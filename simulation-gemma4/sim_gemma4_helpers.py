"""
Gemma-4 PDA Analysis Helpers

Uses HuggingFace hooks (register_forward_hook) for activation extraction.
nnsight can be used as drop-in replacement for more complex interventions later.

Adapted from sim2_helpers.py for Gemma-4 architecture with:
- Per-Layer Embeddings (PLE)
- Shared KV-Cache in later layers
- Alternating local/global attention
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_gemma4(model_id: str = "google/gemma-4-4b-it", quantize_4bit: bool = True):
    """Load Gemma-4 with optional 4-bit quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def extract_activations(model, tokenizer, prompts: List[str], layers: List[int]) -> Dict[int, torch.Tensor]:
    """
    Extract activations from specified layers using forward hooks.
    Processes prompts individually to avoid padding artifacts.
    Returns: Dict[layer] -> [total_tokens, d_model] (2D, no padding)
    """
    all_acts = {l: [] for l in layers}

    for prompt in prompts:
        captured = {}
        hooks = []

        for l in layers:
            layer_module = model.model.layers[l]

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[layer_idx] = hidden.detach().cpu().float()
                return hook_fn

            hooks.append(layer_module.register_forward_hook(make_hook(l)))

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        for l in layers:
            if l in captured:
                act = captured[l]
                if act.ndim == 3:
                    act = act[0]  # [seq, d_model]
                all_acts[l].append(act)

        for h in hooks:
            h.remove()

    return {l: torch.cat(all_acts[l], dim=0) for l in layers if all_acts[l]}


def extract_activations_batched(model, tokenizer, prompts: List[str], layers: List[int]):
    """
    Batch version: all prompts together (with padding).
    Returns: Dict[layer] -> [batch, pos, d_model] (3D), tokens
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    tokens = inputs.input_ids

    captured = {}
    hooks = []

    for l in layers:
        layer_module = model.model.layers[l]

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = hidden.detach().cpu().float()
            return hook_fn

        hooks.append(layer_module.register_forward_hook(make_hook(l)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()
    torch.cuda.empty_cache()

    return captured, tokens


def compute_svd_spectrum(activations: torch.Tensor) -> Dict[str, Any]:
    """SVD spectrum and participation ratio."""
    if activations.ndim == 3:
        flat = activations.reshape(-1, activations.shape[-1])
    else:
        flat = activations

    mean = flat.mean(dim=0)
    centered = flat - mean

    s = torch.linalg.svdvals(centered.float())
    s_sq = s**2
    pr = (s_sq.sum()**2) / (s_sq**2).sum()
    var_exp = s_sq / s_sq.sum()
    cum_var = torch.cumsum(var_exp, dim=0)

    return {
        "singular_values": s.numpy(),
        "participation_ratio": pr.item(),
        "variance_explained": var_exp.numpy(),
        "cumulative_variance_explained": cum_var.numpy(),
    }


def analyze_dominant_direction(acts: torch.Tensor, label: str = "") -> Dict[str, Any]:
    """
    Analyze dominant SVD direction and what remains after removal.
    Core analysis from Sim 2b.
    """
    flat = acts.reshape(-1, acts.shape[-1]).float()
    mean = flat.mean(dim=0)
    centered = flat - mean

    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    dominant_dir = Vh[0]
    total_var = (S**2).sum()
    dom_var = S[0]**2 / total_var

    # Residual after removing dominant direction
    projections = centered @ dominant_dir
    residual = centered - projections.unsqueeze(1) * dominant_dir.unsqueeze(0)

    U_r, S_r, Vh_r = torch.linalg.svd(residual, full_matrices=False)
    total_var_r = (S_r**2).sum()
    cum_var_r = torch.cumsum(S_r**2, dim=0) / total_var_r
    pr_r = (S_r.sum()**2) / (S_r**2).sum()

    pr_orig = (S.sum()**2) / (S**2).sum()
    cum_var_orig = torch.cumsum(S**2, dim=0) / total_var

    n90_orig = int(torch.searchsorted(cum_var_orig, 0.90).item()) + 1
    n90_r = int(torch.searchsorted(cum_var_r, 0.90).item()) + 1

    if label:
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")
        print(f"Dominant direction: {dom_var:.1%} of total variance")
        print(f"SV ratio top1/top2: {S[0]/S[1]:.1f}:1")
        print(f"{'Metric':<25} {'Original':>12} {'After removal':>15}")
        print(f"{'-'*52}")
        print(f"{'Participation Ratio':<25} {pr_orig.item():>12.1f} {pr_r.item():>15.1f}")
        print(f"{'Components for 90%':<25} {n90_orig:>12d} {n90_r:>15d}")

    return {
        "pr_original": pr_orig.item(),
        "pr_residual": pr_r.item(),
        "dominant_var_share": dom_var.item(),
        "sv_original": S.numpy(),
        "sv_residual": S_r.numpy(),
        "cum_var_original": cum_var_orig.numpy(),
        "cum_var_residual": cum_var_r.numpy(),
        "dominant_direction": dominant_dir,
        "mean": mean,
        "n90_orig": n90_orig,
        "n90_residual": n90_r,
    }


def parallel_forward_mean_separated(model, tokenizer, activations, layer_idx, k_subspaces, tokens):
    """
    Mean-separated parallel forward through one layer.
    Adapted from Sim 2b for HuggingFace models.
    """
    original_shape = activations.shape
    flat = activations.reshape(-1, activations.shape[-1]).float()
    mean = flat.mean(dim=0)
    centered = flat - mean

    # SVD
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    # Separate dominant direction
    dominant = U[:, 0:1] @ torch.diag(S[0:1]) @ Vh[0:1, :]
    residual = centered - dominant

    # Residual SVD for subspace decomposition
    U_r, S_r, Vh_r = torch.linalg.svd(residual, full_matrices=False)
    d_model = flat.shape[-1]
    block_size = d_model // k_subspaces

    # Hook names
    hook_name_in = f"model.layers.{layer_idx}"
    hook_name_out = f"model.layers.{layer_idx + 1}"

    outputs = []
    for i in range(k_subspaces):
        start = i * block_size
        end = (i + 1) * block_size if i < k_subspaces - 1 else d_model

        sub_residual = U_r[:, start:end] @ torch.diag(S_r[start:end]) @ Vh_r[start:end, :]
        sub_act = (dominant + sub_residual + mean).reshape(original_shape).to(model.device)

        captured = {}
        hooks = []

        # Inject at layer_idx
        def make_inject(act):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (act,) + output[1:]
                return act
            return hook_fn

        # Capture at layer_idx + 1
        def make_capture(store):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                store["out"] = hidden.detach().cpu().float()
            return hook_fn

        h1 = model.model.layers[layer_idx].register_forward_hook(make_inject(sub_act))
        h2 = model.model.layers[layer_idx + 1].register_forward_hook(make_capture(captured))
        hooks.extend([h1, h2])

        with torch.no_grad():
            model(tokens.to(model.device))

        for h in hooks:
            h.remove()

        if "out" in captured:
            outputs.append(captured["out"])
        torch.cuda.empty_cache()

    return outputs


def get_model_info(model) -> Dict[str, Any]:
    """Extract model architecture info relevant for PDA analysis."""
    config = model.config
    info = {
        "model_type": getattr(config, "model_type", "unknown"),
        "num_layers": config.num_hidden_layers,
        "d_model": config.hidden_size,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
    }

    # Gemma-4 specific
    if hasattr(config, "num_kv_shared_layers"):
        info["num_kv_shared_layers"] = config.num_kv_shared_layers
        info["kv_shared_start"] = info["num_layers"] - config.num_kv_shared_layers
    if hasattr(config, "per_layer_embedding_size"):
        info["ple_size"] = config.per_layer_embedding_size
        info["has_ple"] = True
    else:
        info["has_ple"] = False

    return info
