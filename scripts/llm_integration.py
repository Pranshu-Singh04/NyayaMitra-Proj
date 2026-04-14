"""
llm_integration.py
==================
LLM backends for NyayaMitra.

Classes:
    LLMResponse        ← dataclass for all LLM outputs
    BaseLLM            ← abstract base class
    INLegalLlamaLLM    ← local HuggingFace or Colab ngrok endpoint
    GeminiLLM          ← Google Gemini Flash
    GPT35LLM           ← OpenAI GPT-3.5-turbo

Factory:
    get_llm(model, api_key, colab_url) → BaseLLM

Usage:
    from llm_integration import get_llm
    llm  = get_llm("gemini", api_key="YOUR_KEY")
    resp = llm.generate(system_prompt, user_prompt)
    print(resp.text)
"""

import os
import json
import time
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
# LLM RESPONSE
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class LLMResponse:
    text       : str
    model_name : str
    latency_ms : float
    error      : str = ""
    tokens_used: int = 0

    @property
    def success(self) -> bool:
        return not self.error and bool(self.text.strip())


# ══════════════════════════════════════════════════════════════════════════════
# BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════
class BaseLLM(ABC):
    # IMPROVEMENT 12: task-type → temperature routing
    _TASK_TEMP = {
        "ljp"      : 0.1,   # deterministic — precision over creativity
        "statute"  : 0.1,   # exact statutory text lookup
        "qa"       : 0.3,   # balanced
        "summarise": 0.4,   # slightly creative summaries
    }

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        pass

    def generate_from_prompt(self, prompt) -> LLMResponse:
        """Generate from a Prompt object (from prompt_builder.py)."""
        return self.generate(prompt.system_prompt, prompt.user_prompt)

    def generate_with_task(
        self, system_prompt: str, user_prompt: str, task_type: str = "qa"
    ) -> LLMResponse:
        """
        IMPROVEMENT 12: Generate with a task-specific temperature override.
        Subclasses that support dynamic temperature should override this method.
        Default fallback just calls generate().
        """
        return self.generate(system_prompt, user_prompt)

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# INLegalLlama  (local HuggingFace OR Colab ngrok endpoint)
# ══════════════════════════════════════════════════════════════════════════════
class INLegalLlamaLLM(BaseLLM):
    """
    Two modes:
        "colab"  — POST to ngrok URL from Colab (fast, no local VRAM)
        "local"  — load model directly on this machine (needs ~14GB VRAM)
    """

    DEFAULT_LOCAL_MODEL = "law-ai/InLegalBERT"     # fallback if no Llama available
    LLAMA_MODEL_ID      = "AdapterHub/INLegalLlama-7B-HF"

    def __init__(
        self,
        mode        : str  = "colab",   # "colab" | "local"
        endpoint_url: str  = None,       # ngrok URL for colab mode
        max_new_tokens: int = 512,
        temperature : float = 0.3,
        device      : str  = None,
    ):
        self.mode           = mode
        self.endpoint_url   = endpoint_url
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.device         = device or ("cuda" if self._cuda() else "cpu")
        self._pipe          = None

        if mode == "local":
            self._load_local()

    def _cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_local(self):
        print(f"Loading INLegalLlama locally on {self.device}...")
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch
        tokenizer = AutoTokenizer.from_pretrained(self.LLAMA_MODEL_ID)
        model     = AutoModelForCausalLM.from_pretrained(
            self.LLAMA_MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self._pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, do_sample=True,
        )
        print("INLegalLlama local model ready")

    def _build_llama_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return (
            f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        t0 = time.time()
        if self.mode == "colab":
            return self._generate_colab(system_prompt, user_prompt, t0)
        else:
            return self._generate_local(system_prompt, user_prompt, t0)

    def _generate_colab(self, system_prompt: str, user_prompt: str, t0: float) -> LLMResponse:
        import requests
        if not self.endpoint_url:
            return LLMResponse("", self.model_id, 0,
                               error="endpoint_url not set for colab mode")
        prompt = self._build_llama_prompt(system_prompt, user_prompt)
        try:
            resp = requests.post(
                self.endpoint_url,
                json={"prompt": prompt, "max_new_tokens": self.max_new_tokens,
                      "temperature": self.temperature},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            # Try multiple response field names (Colab endpoints vary)
            text = (
                data.get("generated_text")
                or data.get("text")
                or data.get("response")
                or data.get("output")
                or data.get("result")
                or ""
            )
            # If data is a list (HuggingFace pipeline format), extract first item
            if isinstance(data, list) and data:
                item = data[0]
                text = item.get("generated_text", item.get("text", text))
            text = str(text).strip()
            # Strip the prompt echo if present
            if "[/INST]" in text:
                text = text.split("[/INST]", 1)[-1].strip()
            # Debug: log if response was empty
            if not text:
                print(f"    [INLegalLlama] Empty response. Keys in JSON: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
            return LLMResponse(
                text       = text,
                model_name = self.model_id,
                latency_ms = (time.time() - t0) * 1000,
            )
        except Exception as e:
            return LLMResponse("", self.model_id, (time.time()-t0)*1000, error=str(e))

    def _generate_local(self, system_prompt: str, user_prompt: str, t0: float) -> LLMResponse:
        if not self._pipe:
            return LLMResponse("", self.model_id, 0, error="Model not loaded")
        prompt = self._build_llama_prompt(system_prompt, user_prompt)
        try:
            out  = self._pipe(prompt)[0]["generated_text"]
            text = out[len(prompt):].strip()
            return LLMResponse(
                text       = text,
                model_name = self.model_id,
                latency_ms = (time.time() - t0) * 1000,
            )
        except Exception as e:
            return LLMResponse("", self.model_id, (time.time()-t0)*1000, error=str(e))

    @property
    def model_id(self) -> str:
        return "INLegalLlama-7B"


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI
# ══════════════════════════════════════════════════════════════════════════════
class GeminiLLM(BaseLLM):
    """Google Gemini Flash via google-generativeai SDK."""

    MODEL_NAME   = "gemini-2.0-flash"   # 15 RPM free tier, widely available
    MAX_TOKENS   = 2048
    TEMPERATURE  = 0.3

    def __init__(
        self,
        api_key    : str   = None,
        temperature: float = None,
        max_tokens : int   = None,
    ):
        from google import genai
        from google.genai import types as genai_types
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GOOGLE_API_KEY env var."
            )
        self._client = genai.Client(api_key=key)
        # Disable safety filters for legal content (murder/rape queries get blocked)
        self._safety_settings = [
            genai_types.SafetySetting(
                category  = "HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold = "BLOCK_NONE",
            ),
            genai_types.SafetySetting(
                category  = "HARM_CATEGORY_HARASSMENT",
                threshold = "BLOCK_NONE",
            ),
            genai_types.SafetySetting(
                category  = "HARM_CATEGORY_HATE_SPEECH",
                threshold = "BLOCK_NONE",
            ),
            genai_types.SafetySetting(
                category  = "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold = "BLOCK_NONE",
            ),
        ]
        self._config = genai_types.GenerateContentConfig(
            temperature       = temperature or self.TEMPERATURE,
            max_output_tokens = max_tokens  or self.MAX_TOKENS,
            safety_settings   = self._safety_settings,
        )
        print(f"Gemini ready ({self.MODEL_NAME})")

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return self._generate_with_temp(system_prompt, user_prompt, self._config)

    def generate_with_task(
        self, system_prompt: str, user_prompt: str, task_type: str = "qa"
    ) -> LLMResponse:
        """IMPROVEMENT 12: route to task-specific temperature."""
        from google.genai import types as genai_types
        temp   = self._TASK_TEMP.get(task_type, self.TEMPERATURE)
        config = genai_types.GenerateContentConfig(
            temperature       = temp,
            max_output_tokens = self._config.max_output_tokens,
            safety_settings   = self._safety_settings,
        )
        return self._generate_with_temp(system_prompt, user_prompt, config)

    def _generate_with_temp(self, system_prompt: str, user_prompt: str, config) -> LLMResponse:
        from google.genai import types as genai_types
        t0 = time.time()
        max_retries = 5

        for attempt in range(max_retries):
            try:
                resp = self._client.models.generate_content(
                    model    = self.MODEL_NAME,
                    contents = [
                        genai_types.Content(role="user", parts=[
                            genai_types.Part(text=f"{system_prompt}\n\n{user_prompt}")
                        ])
                    ],
                    config   = config,
                )
                # Try resp.text first; fall back to candidates if empty
                text = resp.text.strip() if resp.text else ""
                if not text and resp.candidates:
                    try:
                        text = resp.candidates[0].content.parts[0].text.strip()
                    except Exception:
                        pass
                if not text and resp.candidates:
                    # Log finish reason so we can diagnose safety blocks
                    try:
                        reason = resp.candidates[0].finish_reason
                        print(f"    [Gemini] empty response — finish_reason={reason}")
                    except Exception:
                        print("    [Gemini] empty response — finish_reason unknown")
                used  = getattr(getattr(resp, "usage_metadata", None), "total_token_count", 0)
                return LLMResponse(
                    text        = text,
                    model_name  = self.model_id,
                    latency_ms  = (time.time() - t0) * 1000,
                    tokens_used = used or 0,
                )
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(k in err_str for k in ["429", "quota", "rate limit", "resource_exhausted"])
                is_server_err = any(k in err_str for k in ["503", "500", "timeout", "deadline", "unavailable"])
                is_retryable  = is_rate_limit or is_server_err

                if is_retryable and attempt < max_retries - 1:
                    # Rate limit: wait much longer — Gemini 2.5 Flash free tier is 10 RPM
                    # Server error: shorter wait, then retry
                    if is_rate_limit:
                        delay = 20.0 * (2 ** attempt)   # 20, 40, 80, 160s
                    else:
                        delay = 8.0  * (2 ** attempt)   # 8, 16, 32, 64s
                    delay = min(delay, 120.0)            # cap at 2 minutes
                    print(f"    [Gemini retry {attempt+1}/{max_retries}]"
                          f" {type(e).__name__} — waiting {delay:.0f}s")
                    time.sleep(delay)
                else:
                    print(f"    [Gemini error] {type(e).__name__}: {str(e)[:150]}")
                    return LLMResponse(
                        "", self.model_id,
                        (time.time() - t0) * 1000,
                        error=str(e)
                    )
        return LLMResponse("", self.model_id,
                           (time.time()-t0)*1000, error="max retries exceeded")

    @property
    def model_id(self) -> str:
        return f"gemini/{self.MODEL_NAME}"


# ══════════════════════════════════════════════════════════════════════════════
# GPT-3.5
# ══════════════════════════════════════════════════════════════════════════════
class GPT35LLM(BaseLLM):
    """OpenAI GPT-3.5-turbo via openai SDK."""

    MODEL_NAME  = "gpt-4o-mini"
    MAX_TOKENS  = 2048
    TEMPERATURE = 0.3

    def __init__(
        self,
        api_key    : str   = None,
        temperature: float = None,
        max_tokens : int   = None,
    ):
        import openai
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Pass api_key= or set OPENAI_API_KEY env var."
            )
        self._client     = openai.OpenAI(api_key=key)
        self._temperature = temperature or self.TEMPERATURE
        self._max_tokens  = max_tokens  or self.MAX_TOKENS
        print(f"GPT-3.5 ready ({self.MODEL_NAME})")

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return self._generate_with_temp(system_prompt, user_prompt, self._temperature)

    def generate_with_task(
        self, system_prompt: str, user_prompt: str, task_type: str = "qa"
    ) -> LLMResponse:
        """IMPROVEMENT 12: route to task-specific temperature."""
        temp = self._TASK_TEMP.get(task_type, self._temperature)
        return self._generate_with_temp(system_prompt, user_prompt, temp)

    def _generate_with_temp(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        t0 = time.time()
        max_retries = 4
        base_delay  = 2.0

        for attempt in range(max_retries):
            try:
                resp  = self._client.chat.completions.create(
                    model       = self.MODEL_NAME,
                    messages    = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature = temperature,
                    max_tokens  = self._max_tokens,
                )
                text   = resp.choices[0].message.content.strip()
                tokens = resp.usage.total_tokens if resp.usage else 0
                return LLMResponse(
                    text        = text,
                    model_name  = self.model_id,
                    latency_ms  = (time.time() - t0) * 1000,
                    tokens_used = tokens,
                )
            except Exception as e:
                err_str = str(e).lower()
                is_retryable = any(
                    k in err_str for k in
                    ["429", "rate_limit", "timeout", "503", "500", "overloaded"]
                )
                if is_retryable and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"    [GPT retry {attempt+1}/{max_retries}]"
                          f" {type(e).__name__} — waiting {delay:.0f}s")
                    time.sleep(delay)
                else:
                    return LLMResponse("", self.model_id, (time.time()-t0)*1000, error=str(e))
        return LLMResponse("", self.model_id,
                           (time.time()-t0)*1000, error="max retries exceeded")

    @property
    def model_id(self) -> str:
        return f"openai/{self.MODEL_NAME}"


# ══════════════════════════════════════════════════════════════════════════════
# GROQ  (free tier — Llama 3.3 70B via OpenAI-compatible API)
# ══════════════════════════════════════════════════════════════════════════════
class GroqLLM(BaseLLM):
    """
    Groq Cloud — free tier, OpenAI-compatible API.
    Get a free key at: https://console.groq.com
    Models: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
    """

    MODEL_NAME  = "llama-3.1-8b-instant"   # 20k TPM (vs 6k for 70b) — needed for long LJP prompts
    MAX_TOKENS  = 1024
    TEMPERATURE = 0.3
    BASE_URL    = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key    : str   = None,
        temperature: float = None,
        max_tokens : int   = None,
    ):
        import openai
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "Groq API key required. Pass api_key= or set GROQ_API_KEY env var. "
                "Get a free key at https://console.groq.com"
            )
        self._client      = openai.OpenAI(api_key=key, base_url=self.BASE_URL)
        self._temperature = temperature or self.TEMPERATURE
        self._max_tokens  = max_tokens  or self.MAX_TOKENS
        print(f"Groq ready ({self.MODEL_NAME})")

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return self._generate_with_temp(system_prompt, user_prompt, self._temperature)

    def generate_with_task(
        self, system_prompt: str, user_prompt: str, task_type: str = "qa"
    ) -> LLMResponse:
        temp = self._TASK_TEMP.get(task_type, self._temperature)
        return self._generate_with_temp(system_prompt, user_prompt, temp)

    def _generate_with_temp(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        t0 = time.time()
        max_retries = 4
        base_delay  = 2.0

        for attempt in range(max_retries):
            try:
                resp  = self._client.chat.completions.create(
                    model       = self.MODEL_NAME,
                    messages    = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature = temperature,
                    max_tokens  = self._max_tokens,
                )
                choice = resp.choices[0]
                text   = (choice.message.content or "").strip()
                tokens = resp.usage.total_tokens if resp.usage else 0

                # Log finish reason if response is empty
                if not text:
                    reason = getattr(choice, "finish_reason", "unknown")
                    print(f"    [Groq] empty response — finish_reason={reason}")

                return LLMResponse(
                    text        = text,
                    model_name  = self.model_id,
                    latency_ms  = (time.time() - t0) * 1000,
                    tokens_used = tokens,
                )
            except Exception as e:
                err_str = str(e).lower()
                is_retryable = any(
                    k in err_str for k in
                    ["429", "rate", "timeout", "503", "500",
                     "overloaded", "unavailable", "deadline"]
                )
                if is_retryable and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"    [Groq retry {attempt+1}/{max_retries}]"
                          f" {type(e).__name__} — waiting {delay:.0f}s")
                    time.sleep(delay)
                else:
                    print(f"    [Groq error] {type(e).__name__}: {str(e)[:150]}")
                    return LLMResponse("", self.model_id,
                                       (time.time()-t0)*1000, error=str(e))
        return LLMResponse("", self.model_id,
                           (time.time()-t0)*1000, error="max retries exceeded")

    @property
    def model_id(self) -> str:
        return f"groq/{self.MODEL_NAME}"


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════
def get_llm(
    model     : str  = "gemini",
    api_key   : str  = None,
    colab_url : str  = None,
    **kwargs
) -> BaseLLM:
    """
    Factory — get any LLM with one line.

    Args:
        model    : "gemini" | "gpt" | "groq" | "inlegalllama"
        api_key  : API key (or set env var GOOGLE_API_KEY / OPENAI_API_KEY / GROQ_API_KEY)
        colab_url: ngrok URL for INLegalLlama on Colab

    Examples:
        llm = get_llm("gemini",       api_key="AIza...")
        llm = get_llm("groq",         api_key="gsk_...")   # free
        llm = get_llm("gpt",          api_key="sk-...")
        llm = get_llm("inlegalllama", colab_url="https://xxx.ngrok-free.app/generate")
    """
    model = model.lower().strip()

    if model in ("gemini", "gemini-flash", "gemini2.5"):
        return GeminiLLM(api_key=api_key, **kwargs)

    elif model in ("gpt", "gpt4", "gpt-4o", "openai"):
        return GPT35LLM(api_key=api_key, **kwargs)

    elif model in ("groq", "llama", "llama3"):
        return GroqLLM(api_key=api_key, **kwargs)

    elif model in ("inlegalllama", "legal-llama"):
        mode = "colab" if colab_url else "local"
        return INLegalLlamaLLM(mode=mode, endpoint_url=colab_url, **kwargs)

    else:
        raise ValueError(
            f"Unknown model: '{model}'. "
            "Choose from: 'gemini', 'gpt', 'groq', 'inlegalllama'"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI  (python llm_integration.py --model gemini --api_key ...)
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Test an LLM backend")
    parser.add_argument("--model",     default="gemini",
                        choices=["gemini", "gpt", "groq", "inlegalllama"])
    parser.add_argument("--api_key",   default=None)
    parser.add_argument("--colab_url", default=None)
    parser.add_argument("--query",     default="What is Section 302 IPC?")
    args = parser.parse_args()

    llm  = get_llm(args.model, api_key=args.api_key, colab_url=args.colab_url)
    sys_ = "You are NyayaMitra, an AI legal advisor for Indian law."
    resp = llm.generate(sys_, args.query)

    print(f"\nModel     : {resp.model_name}")
    print(f"Latency   : {resp.latency_ms:.0f}ms")
    print(f"Tokens    : {resp.tokens_used}")
    print(f"Success   : {resp.success}")
    if resp.error:
        print(f"Error     : {resp.error}")
    print(f"\nResponse:\n{resp.text}")


if __name__ == "__main__":
    main()