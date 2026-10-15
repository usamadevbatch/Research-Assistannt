"""Pluggable LLM backends with streaming."""
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator

import requests


class LLM(ABC):
    @abstractmethod
    def stream(self, system: str, messages: list[dict]) -> Iterator[str]: ...

    def complete(self, system: str, messages: list[dict]) -> str:
        return "".join(self.stream(system, messages))

    def complete_json(self, system: str, messages: list[dict]):
        """Ask for a JSON answer and parse it, tolerating code fences or extra prose around it.

        Small local models occasionally loop or truncate; retry once with a reminder to stay compact.
        """
        try:
            return parse_json(self._complete_json(system, messages))
        except ValueError:
            reminder = "\n\nKeep every list short (at most 6 items, no repeats) and output compact, complete JSON."
            return parse_json(self._complete_json(system + reminder, messages))

    def _complete_json(self, system: str, messages: list[dict]) -> str:
        return self.complete(system + "\n\nRespond with valid JSON only.", messages)


def parse_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            pass
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"Model did not return JSON: {text[:200]}")


class GeminiLLM(LLM):
    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def stream(self, system, messages):
        t = self._types
        contents = [
            t.Content(role="user" if m["role"] == "user" else "model", parts=[t.Part(text=m["content"])])
            for m in messages
        ]
        config = t.GenerateContentConfig(system_instruction=system, temperature=0.2)
        for chunk in self._client.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        ):
            if chunk.text:
                yield chunk.text

    def _complete_json(self, system, messages):
        t = self._types
        contents = [
            t.Content(role="user" if m["role"] == "user" else "model", parts=[t.Part(text=m["content"])])
            for m in messages
        ]
        config = t.GenerateContentConfig(system_instruction=system, temperature=0.1,
                                         response_mime_type="application/json")
        return self._client.models.generate_content(model=self.model, contents=contents, config=config).text


class OllamaLLM(LLM):
    def __init__(self, base_url: str, model: str, num_ctx: int = 8192):
        self.base_url = base_url
        self.model = model
        self.num_ctx = num_ctx  # Ollama's default context is too small for RAG prompts

    def stream(self, system, messages):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
            # num_predict caps answer length so a CPU-bound model can't ramble for minutes.
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx, "num_predict": 1200},
            "keep_alive": "30m",  # avoid reloading the model from disk between questions
        }
        with requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                if content := data.get("message", {}).get("content"):
                    yield content
                if data.get("done"):
                    break


    def _complete_json(self, system, messages):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "format": "json",
            # The repeat penalty stops small models from looping on one list item until the length cap.
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx, "num_predict": 1800,
                        "repeat_penalty": 1.15, "repeat_last_n": 256},
            "keep_alive": "30m",
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=1200)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return data.get("message", {}).get("content", "")


def ollama_models(base_url: str) -> list[str]:
    """Models already pulled into the local Ollama, or [] if it isn't running."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


def make_llm(backend: str, *, api_key: str, model: str, ollama_url: str) -> LLM:
    if backend.startswith("Gemini"):
        return GeminiLLM(api_key, model)
    return OllamaLLM(ollama_url, model)
