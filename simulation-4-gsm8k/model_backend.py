"""Pluggable model backend for Sim 4 / Sim 4b.

Two backends supported:

  openrouter (default)
    Calls Qwen3-8B via OpenRouter's OpenAI-compatible API.
    Needs OPENROUTER_API_KEY env var, or a token at
    ~/.config/api-keys/openrouter.

  local_hf
    Loads Qwen/Qwen3-8B locally via transformers.AutoModelForCausalLM
    in bf16 on the first available CUDA device. Reuses one process-level
    pipeline so the model is loaded once even when called many times.
    Needs torch + transformers (in requirements-local.txt).

Select with environment variable PDA_BACKEND:
  PDA_BACKEND=openrouter  (default)
  PDA_BACKEND=local_hf

Both backends expose the same signature:
  call_model(system: str, user: str, temperature: float = 0.7,
             max_tokens: int = 1024) -> str

The local backend is what you want when running on Modal / RunPod / a
GPU pod with no API budget. The OpenRouter backend is what was used to
produce the saved result artifacts in this repo (Apr 2026). The two
backends will not produce byte-identical answers -- different sampling
implementations, different model snapshots -- but should produce
qualitatively comparable accuracy.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND = os.environ.get("PDA_BACKEND", "openrouter").lower()
OPENROUTER_MODEL = os.environ.get("PDA_OPENROUTER_MODEL", "qwen/qwen3-8b")
LOCAL_MODEL = os.environ.get("PDA_LOCAL_MODEL", "Qwen/Qwen3-8B")


# --- OpenRouter backend ---------------------------------------------------

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        key_file = Path(os.path.expanduser("~/.config/api-keys/openrouter"))
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        raise RuntimeError(
            "OpenRouter backend requires OPENROUTER_API_KEY env var "
            "or ~/.config/api-keys/openrouter file."
        )
    _openai_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return _openai_client


def _call_openrouter(system: str, user: str, temperature: float, max_tokens: int) -> str:
    client = _get_openai_client()
    try:
        r = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"  API error: {e}")
        return ""


# --- Local HuggingFace backend --------------------------------------------

_hf_model = None
_hf_tokenizer = None


def _load_hf():
    global _hf_model, _hf_tokenizer
    if _hf_model is not None:
        return _hf_model, _hf_tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[backend:local_hf] loading {LOCAL_MODEL} (bf16, device=cuda)...", flush=True)
    _hf_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, trust_remote_code=True)
    _hf_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    _hf_model.eval()
    print(f"[backend:local_hf] loaded.", flush=True)
    return _hf_model, _hf_tokenizer


def _call_local_hf(system: str, user: str, temperature: float, max_tokens: int) -> str:
    import torch
    model, tok = _load_hf()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=0.95 if do_sample else 1.0,
            pad_token_id=tok.eos_token_id,
        )
    response = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# --- Public API -----------------------------------------------------------


def call_model(system: str, user: str, temperature: float = 0.7,
               max_tokens: int = 1024) -> str:
    if BACKEND == "openrouter":
        return _call_openrouter(system, user, temperature, max_tokens)
    elif BACKEND == "local_hf":
        return _call_local_hf(system, user, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown PDA_BACKEND: {BACKEND!r}; expected 'openrouter' or 'local_hf'")


def describe_backend() -> str:
    """Return a one-line description suitable for the run config block."""
    if BACKEND == "openrouter":
        return f"openrouter:{OPENROUTER_MODEL}"
    return f"local_hf:{LOCAL_MODEL}"
