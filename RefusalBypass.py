"""
title: Refusal Bypass
author: erbmur
author_url: https://github.com/erbmur
version: 1.4
license: MIT
"""

import requests
import json
import time
import asyncio
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Callable, Awaitable


class Filter:
    class Valves(BaseModel):
        # The model to use when the main model refuses
        abliterated_model_id: str = Field(
            default="dolphin-mistral",
            description="The ID of the abliterated/uncensored model to load for refusals.",
        )
        # The model used to classify if a response is a refusal
        judge_model_id: str = Field(
            default="llama3.2",
            description="Small, fast model used to classify if the response is a refusal (e.g., llama3.2, gemma:2b).",
        )
        # The URL of your Ollama instance (usually localhost:11434)
        ollama_base_url: str = Field(
            default="http://host.docker.internal:11434",
            description="The base URL for the Ollama API (use host.docker.internal if running Open WebUI in Docker).",
        )
        debug: bool = Field(
            default=False, description="Enable debug printing in the console."
        )

    def __init__(self):
        self.valves = self.Valves()

    def check_refusal(self, content: str) -> bool:
        """
        Uses an LLM Judge to check if the content is a refusal.
        Blocking call - should be run in executor.
        """
        # Quick check for empty content
        if not content or not content.strip():
            return False

        url = f"{self.valves.ollama_base_url}/api/generate"

        # We limit context to first 1000 chars to speed up judging
        context_preview = content[:1000]

        prompt = (
            f"Analyze the following AI response text. Does it represent a refusal to answer a request "
            f"due to ethical, safety, moral, or policy guidelines? "
            f"Typical refusals sound like 'I cannot fulfill', 'I apologize', 'I cannot comply'.\n\n"
            f'Text: "{context_preview}"\n\n'
            f"Answer with exactly one word: 'YES' or 'NO'."
        )

        payload = {
            "model": self.valves.judge_model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 10},  # We only need a yes/no
        }

        try:
            if self.valves.debug:
                print(
                    f"DEBUG: asking {self.valves.judge_model_id} to judge response..."
                )

            response = requests.post(url, json=payload)
            response.raise_for_status()

            result = response.json().get("response", "").strip().upper()

            if self.valves.debug:
                print(f"DEBUG: Judge verdict: {result}")

            return "YES" in result

        except Exception as e:
            if self.valves.debug:
                print(f"Error checking refusal with judge: {e}")
            # Fail safe: return False so we don't loop or break
            return False

    def unload_model(self, model_id: str):
        """Unloads a model from memory. Blocking call."""
        url = f"{self.valves.ollama_base_url}/api/generate"
        payload = {"model": model_id, "keep_alive": 0}
        try:
            requests.post(url, json=payload)
            if self.valves.debug:
                print(f"DEBUG: Unloaded {model_id}")
        except Exception as e:
            print(f"Error unloading {model_id}: {e}")

    def load_model(self, model_id: str):
        """Preloads a model into memory. Blocking call."""
        url = f"{self.valves.ollama_base_url}/api/generate"
        payload = {"model": model_id, "keep_alive": -1}
        try:
            requests.post(url, json=payload)
            if self.valves.debug:
                print(f"DEBUG: Reloaded {model_id}")
        except Exception as e:
            print(f"Error loading {model_id}: {e}")

    def generate_replacement(
        self, model_id: str, messages: List[Dict[str, Any]]
    ) -> str:
        """Generates a response using the abliterated model. Blocking call."""
        url = f"{self.valves.ollama_base_url}/api/chat"

        # We only need the message history
        payload = {"model": model_id, "messages": messages, "stream": False}

        try:
            if self.valves.debug:
                print(f"DEBUG: Generating with {model_id}...")
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get(
                "content", "Error generating replacement."
            )
        except Exception as e:
            return f"System Error: Failed to swap models. {e}"

    async def emit_status(self, emitter: Callable, message: str, done: bool = False):
        """Helper to emit status updates to the UI."""
        if emitter:
            await emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )

    async def outlet(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> dict:
        """
        The outlet function processes the response AFTER the LLM has finished.
        """
        messages = body.get("messages", [])
        if not messages:
            return body

        ai_response = messages[-1].get("content", "")
        original_model = body.get("model")

        # Get the running event loop to offload blocking tasks
        loop = asyncio.get_running_loop()

        # 1. Check for refusal (Offload to thread to prevent UI blocking during potential model load)
        await self.emit_status(
            __event_emitter__, "Checking response for refusal...", False
        )

        try:
            # Run the synchronous check_refusal in a separate thread
            is_refusal = await loop.run_in_executor(
                None, self.check_refusal, ai_response
            )
        except Exception as e:
            print(f"Error in async execution of check_refusal: {e}")
            is_refusal = False

        if is_refusal:
            print(f"Refusal detected from {original_model}. Initiating swap sequence.")
            await self.emit_status(
                __event_emitter__,
                f"Refusal detected. Unloading {original_model}...",
                False,
            )

            history_minus_refusal = messages[:-1]

            # 2. Unload Original Model (Async offload)
            await loop.run_in_executor(None, self.unload_model, original_model)

            # 3. Generate with Abliterated Model (Async offload)
            await self.emit_status(
                __event_emitter__,
                f"Loading {self.valves.abliterated_model_id} & Generating...",
                False,
            )

            new_answer = await loop.run_in_executor(
                None,
                self.generate_replacement,
                self.valves.abliterated_model_id,
                history_minus_refusal,
            )

            # 4. Unload Abliterated Model (Async offload)
            await self.emit_status(
                __event_emitter__,
                f"Unloading {self.valves.abliterated_model_id}...",
                False,
            )
            await loop.run_in_executor(
                None, self.unload_model, self.valves.abliterated_model_id
            )

            # 5. Reload Original Model (Async offload)
            await self.emit_status(
                __event_emitter__, f"Restoring {original_model}...", False
            )
            await loop.run_in_executor(None, self.load_model, original_model)

            await self.emit_status(__event_emitter__, "Model swap complete.", True)

            # 6. Update the body with the new answer
            body["messages"][-1]["content"] = new_answer

        else:
            # Clear status if no refusal
            await self.emit_status(__event_emitter__, "", True)

        return body