"""
core/llm_client.py

LLM abstraction layer. Swap backends without touching experiment code.

Supported backends:
  - dummy                         — seeded deterministic mock (no API needed)
  - openrouter/<model>            — OpenRouter API (set OPENROUTER_API_KEY)
  - ollama/<model>                — Local Ollama instance
  - nvidia/<model>                — NVIDIA API (set NVIDIA_API_KEY)
  - groq/<model>                  — Groq API (set GROQ_API_KEY) 

Usage:
  client = LLMClient.from_model_string("dummy", seed=42)
  client = LLMClient.from_model_string("openrouter/meta-llama/llama-3.1-8b-instruct:free")
  client = LLMClient.from_model_string("ollama/llama3.2")
  client = LLMClient.from_model_string("nvidia/mistralai/mistral-medium-3.5-128b")
"""

from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()
import requests
import re
from groq import Groq

from utils.cache import get_cached, make_cache_key, set_cached


class LLMClient(ABC):
    """Abstract LLM client. All backends implement generate()."""

    @abstractmethod
    def generate(
        self,
        messages: List[dict],
        model: str,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: OpenAI-style [{"role": "...", "content": "..."}]
            model: model identifier string
            temperature: sampling temperature

        Returns:
            Raw response string (expected to be JSON for agents)
        """

    @classmethod
    def from_model_string(cls, model_string: str, seed: int = 42) -> "LLMClient":
        """
        Factory: parse model string and return appropriate backend.

        Format:
          "dummy"                     → DummyClient
          "openrouter/<model>"        → OpenRouterClient
          "ollama/<model>"            → OllamaClient
          "nvidia/<model>"            → NVIDIAClient
          "groq/<model>"              → GroqClient  
        """
        if model_string == "dummy":
            return DummyClient(seed=seed)
        elif model_string.startswith("openrouter/"):
            model = model_string[len("openrouter/"):]
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable not set. "
                    "Export it or use --model dummy for testing."
                )
            return OpenRouterClient(api_key=api_key, default_model=model)
        elif model_string.startswith("ollama/"):
            model = model_string[len("ollama/"):]
            return OllamaClient(default_model=model)
        elif model_string.startswith("nvidia/"):
            model = model_string[len("nvidia/"):]
            api_key = os.environ.get("NVIDIA_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "NVIDIA_API_KEY environment variable not set. "
                    "Export it or use --model dummy for testing."
                )
            return NVIDIAClient(api_key=api_key, default_model=model)
        elif model_string.startswith("groq/"):
            # Remove the "groq/" prefix to get the actual model name
            model = model_string[len("groq/"):]
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY environment variable not set. "
                    "Export it or use --model dummy for testing."
                )
            return GroqClient(api_key=api_key, default_model=model)
        else:
            raise ValueError(
                f"Unknown model string: {model_string!r}. "
                "Use 'dummy', 'openrouter/<model>', 'ollama/<model>', 'nvidia/<model>', or 'groq/<model>'."
            )


# Dummy client — deterministic seeded mock, no API costs

DUMMY_CLAIMS = [
    "Vitamin D supplementation reduces the risk of acute respiratory infections.",
    "Vitamin D does not significantly affect respiratory infection rates.",
    "The evidence on vitamin D and immunity is inconclusive.",
]

DUMMY_REASONINGS = [
    "Based on the meta-analysis in D2, the protective effect is well-established.",
    "The evidence from D2 has been retracted and cannot be relied upon.",
    "Documents D1 and D3 support immune function but do not directly address infection risk reduction.",
]


class DummyClient(LLMClient):
    """
    Seeded deterministic mock LLM for testing and debugging.
    Returns valid JSON messages without any API call.
    Behaviour is fully reproducible given the same seed.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(
        self,
        messages: List[dict],
        model: str = "dummy",
        temperature: float = 0.2,
    ) -> str:
        # Deterministic choice based on prompt hash
        key = make_cache_key(model, messages, temperature)
        cached = get_cached(key)
        if cached:
            return cached

        # Extract agent_id and round from system/user content heuristically
        content_blob = " ".join(m.get("content", "") for m in messages)
        agent_id = "A?"
        round_num = 0
        for token in content_blob.split():
            if token.startswith("A") and token[1:].isdigit():
                agent_id = token
            if token.isdigit() and 0 < int(token) <= 100:
                round_num = int(token)

        # Pick claim + reasoning based on seeded random
        claim_idx = self._rng.randint(0, len(DUMMY_CLAIMS) - 1)
        reason_idx = self._rng.randint(0, len(DUMMY_REASONINGS) - 1)
        confidence = round(self._rng.uniform(0.3, 0.95), 2)
        msg_type = self._rng.choice(["support", "contradict", "question"])

        response = json.dumps({
            "claim": DUMMY_CLAIMS[claim_idx],
            "confidence": confidence,
            "evidence_ids": self._rng.sample(["D1", "D2", "D3", "D4", "D5"], k=2),
            "reasoning": DUMMY_REASONINGS[reason_idx],
            "type": msg_type,
            "source_agent": agent_id,
            "round": round_num,
        })

        set_cached(key, response)
        return response


# OpenRouter client 
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_SYSTEM_PROMPT = """You are an AI agent participating in a multi-agent discussion about scientific claims.

CRITICAL INSTRUCTION: You MUST respond with ONLY a valid JSON object. Do not include any other text, explanations, or markdown.

The JSON MUST have these exact fields:
{
    "claim": "The specific claim you are addressing",
    "confidence": 0.75,
    "evidence_ids": ["D1", "D3"],
    "reasoning": "Your detailed reasoning for this position",
    "type": "support"  // Must be exactly: "support", "contradict", or "question"
}

The 'type' field MUST be exactly one of: "support", "contradict", or "question".
Your response must be ONLY the JSON object, nothing else."""


class OpenRouterClient(LLMClient):
    """OpenRouter API client with response caching."""

    def __init__(self, api_key: str, default_model: str):
        self.api_key = api_key
        self.default_model = default_model

    def _extract_and_validate_json(self, text: str) -> str:
        """Extract JSON from text and ensure it has all required fields."""
        # Try to find JSON in the text
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        
        # Try each match
        for match in matches:
            try:
                parsed = json.loads(match)
                # Check if it has the required fields
                if all(key in parsed for key in ['claim', 'confidence', 'type']):
                    # Ensure type is valid
                    if parsed['type'] not in ['support', 'contradict', 'question']:
                        parsed['type'] = 'question'
                    # Ensure confidence is a float between 0 and 1
                    if not isinstance(parsed['confidence'], (int, float)):
                        parsed['confidence'] = 0.5
                    parsed['confidence'] = max(0.0, min(1.0, float(parsed['confidence'])))
                    # Ensure evidence_ids is a list
                    if 'evidence_ids' not in parsed or not isinstance(parsed['evidence_ids'], list):
                        parsed['evidence_ids'] = []
                    # Ensure reasoning exists
                    if 'reasoning' not in parsed:
                        parsed['reasoning'] = "No reasoning provided."
                    return json.dumps(parsed)
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, create a fallback
        return json.dumps({
            "claim": "Unable to parse response",
            "confidence": 0.5,
            "evidence_ids": [],
            "reasoning": "The model did not return valid JSON. Using fallback.",
            "type": "question"
        })

    def generate(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 16384,
        top_p: float = 1.0,
    ) -> str:
        model = model or self.default_model
        key = make_cache_key(model, messages, temperature)

        cached = get_cached(key)
        if cached:
            return cached

        # Ensure we have a system prompt that enforces JSON output
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            # Add system prompt at the beginning
            messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + messages
        else:
            # Update existing system prompt to include JSON requirement
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    messages[i]["content"] = DEFAULT_SYSTEM_PROMPT + "\n\n" + msg["content"]
                    break

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/multiagent-sim",
            "X-Title": "MultiAgent-LLM-Sim",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            
            response_data = resp.json()
            
            # Check if response has choices
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                raise ValueError(f"Unexpected response format: {response_data}")
            
            content = response_data["choices"][0]["message"]["content"]
            
            # Extract and validate JSON
            content = self._extract_and_validate_json(content)
            
            set_cached(key, content)
            return content
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter API request failed: {e}") from e
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"OpenRouter API response parsing failed: {e}") from e


# Ollama client (local)
OLLAMA_URL = "http://localhost:11434/api/chat"


class OllamaClient(LLMClient):
    """Local Ollama client with response caching."""

    def __init__(self, default_model: str, base_url: str = OLLAMA_URL):
        self.default_model = default_model
        self.base_url = base_url

    def generate(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        model = model or self.default_model
        key = make_cache_key(model, messages, temperature)

        cached = get_cached(key)
        if cached:
            return cached

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
            "format": "json",
        }

        resp = requests.post(self.base_url, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]

        set_cached(key, content)
        return content

# NVIDIA client
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


class NVIDIAClient(LLMClient):
    """NVIDIA API client with response caching."""

    def __init__(self, api_key: str, default_model: str):
        self.api_key = api_key
        self.default_model = default_model

    def generate(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 16384,
        top_p: float = 1.0,
    ) -> str:
        model = model or self.default_model
        key = make_cache_key(model, messages, temperature)

        cached = get_cached(key)
        if cached:
            return cached

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        
        # Add reasoning_effort for models that support it (like Mistral)
        # Only add if the model supports it, otherwise it might cause errors
        if "mistral" in model.lower():
            payload["reasoning_effort"] = "high"

        resp = requests.post(NVIDIA_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        
        # Parse the response
        response_data = resp.json()
        content = response_data["choices"][0]["message"]["content"]
        
        # Try to ensure the response is valid JSON
        # If it's not JSON, wrap it in a JSON structure to maintain compatibility
        try:
            json.loads(content)
        except json.JSONDecodeError:
            # If the response isn't valid JSON, wrap it
            content = json.dumps({
                "content": content,
                "type": "response",
            })

        set_cached(key, content)
        return content

# ---------------------------------------------------------------------------
# Groq client (fast inference using official SDK)
# ---------------------------------------------------------------------------

class GroqClient(LLMClient):
    """Groq API client using the official SDK with response caching."""
    
    def __init__(self, api_key: str, default_model: str):
        self.api_key = api_key
        self.default_model = default_model
        # Import groq here to avoid dependency issues
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Groq SDK not installed. Run: pip install groq"
            )
    
    def _extract_and_validate_json(self, text: str) -> str:
        """Extract JSON from text and ensure it has all required fields."""
        # Try to find JSON in the text
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        
        # Try each match
        for match in matches:
            try:
                parsed = json.loads(match)
                # Check if it has the required fields
                if all(key in parsed for key in ['claim', 'confidence', 'type']):
                    # Ensure type is valid
                    if parsed['type'] not in ['support', 'contradict', 'question']:
                        parsed['type'] = 'question'
                    # Ensure confidence is a float between 0 and 1
                    if not isinstance(parsed['confidence'], (int, float)):
                        parsed['confidence'] = 0.5
                    parsed['confidence'] = max(0.0, min(1.0, float(parsed['confidence'])))
                    # Ensure evidence_ids is a list
                    if 'evidence_ids' not in parsed or not isinstance(parsed['evidence_ids'], list):
                        parsed['evidence_ids'] = []
                    # Ensure reasoning exists
                    if 'reasoning' not in parsed:
                        parsed['reasoning'] = "No reasoning provided."
                    return json.dumps(parsed)
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, create a fallback
        return json.dumps({
            "claim": "Unable to parse response",
            "confidence": 0.5,
            "evidence_ids": [],
            "reasoning": "The model did not return valid JSON. Using fallback.",
            "type": "question"
        })
    
    def generate(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        top_p: float = 1.0,
    ) -> str:
        # Use the model directly (no prefix)
        model = model or self.default_model
        
        # IMPORTANT: Make sure we don't have any "groq/" prefix
        if model.startswith("groq/"):
            model = model[len("groq/"):]
        
        key = make_cache_key(model, messages, temperature)
        
        cached = get_cached(key)
        if cached:
            return cached
        
        # Ensure we have a system prompt that enforces JSON output
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + messages
        
        try:
            # Use the Groq SDK with the model name directly
            print(f"[DEBUG] Calling Groq with model: {model}")  # Debug line
            
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=model,  # This should be just the model name, e.g., "llama-3.3-70b-versatile"
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                response_format={"type": "json_object"},
            )
            
            content = chat_completion.choices[0].message.content
            
            # Extract and validate JSON
            content = self._extract_and_validate_json(content)
            
            set_cached(key, content)
            return content
            
        except Exception as e:
            error_msg = f"Groq API request failed: {e}"
            raise RuntimeError(error_msg) from e