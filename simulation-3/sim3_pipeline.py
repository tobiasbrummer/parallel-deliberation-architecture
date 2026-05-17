"""
Simulation 3: Multi-Layer PDA Pipeline with Mean-Separation

Core idea from Sim 2b: A single dominant SVD direction masks all subspace structure.
Removing it (like DC-offset in audio) reveals rich, separable subspaces.

This module implements the full pipeline for running PDA on existing models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformer_lens import HookedTransformer
from dataclasses import dataclass


@dataclass
class DominantDirection:
    """Cached dominant direction for a specific layer."""
    direction: torch.Tensor      # [d_model] - unit vector
    mean: torch.Tensor           # [d_model] - activation mean
    variance_share: float        # fraction of total variance
    sv_ratio: float              # top1/top2 singular value ratio


@dataclass
class PDAResult:
    """Result of a PDA forward pass through one or more layers."""
    merged_activations: torch.Tensor   # [batch, pos, d_model]
    original_activations: torch.Tensor # [batch, pos, d_model] - ground truth
    cos_sim: float
    mse: float
    n_layers: int
    k_subspaces: int
    layer_range: Tuple[int, int]
    per_layer_cos_sim: List[float]     # degradation curve


def compute_dominant_direction(activations: torch.Tensor) -> DominantDirection:
    """
    Extract the dominant SVD direction from activations.

    Args:
        activations: [n_tokens, d_model] or [batch, pos, d_model]

    Returns:
        DominantDirection with cached direction, mean, and stats
    """
    flat = activations.reshape(-1, activations.shape[-1]).float()
    mean = flat.mean(dim=0)
    centered = flat - mean

    # We only need top-2 singular values for the direction and ratio
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    dominant_dir = Vh[0]  # top-1 right singular vector
    variance_share = (S[0]**2 / (S**2).sum()).item()
    sv_ratio = (S[0] / S[1]).item() if S[1] > 0 else float('inf')

    return DominantDirection(
        direction=dominant_dir,
        mean=mean,
        variance_share=variance_share,
        sv_ratio=sv_ratio,
    )


def separate_dominant(activations: torch.Tensor, dom: DominantDirection) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Separate activations into dominant component and residual.

    Returns:
        (dominant_component, residual) - both same shape as input
    """
    flat = activations.reshape(-1, activations.shape[-1]).float()
    centered = flat - dom.mean

    # Project onto dominant direction
    projections = centered @ dom.direction  # [n_tokens]
    dominant_component = projections.unsqueeze(-1) * dom.direction.unsqueeze(0)  # [n_tokens, d_model]
    residual = centered - dominant_component

    # Reshape back
    dominant_component = (dominant_component + dom.mean).reshape(activations.shape)
    residual = residual.reshape(activations.shape)

    return dominant_component, residual


def decompose_residual_subspaces(residual: torch.Tensor, k: int) -> List[torch.Tensor]:
    """
    Decompose the residual (after dominant removal) into k SVD-based subspaces.

    Args:
        residual: [batch, pos, d_model] or [n_tokens, d_model]
        k: number of subspaces

    Returns:
        List of k tensors, each containing the reconstruction from that subspace block
    """
    original_shape = residual.shape
    flat = residual.reshape(-1, residual.shape[-1]).float()

    U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
    d_model = flat.shape[-1]
    block_size = d_model // k

    subspaces = []
    for i in range(k):
        start = i * block_size
        end = (i + 1) * block_size if i < k - 1 else d_model

        # Reconstruct this block
        sub = U[:, start:end] @ torch.diag(S[start:end]) @ Vh[start:end, :]
        subspaces.append(sub.reshape(original_shape))

    return subspaces


def pda_forward_single_layer(
    model: HookedTransformer,
    activations: torch.Tensor,
    layer_idx: int,
    k: int,
    tokens: torch.Tensor,
    dom: Optional[DominantDirection] = None,
    recompute_dominant: bool = True,
) -> Tuple[torch.Tensor, DominantDirection, Dict[str, Any]]:
    """
    PDA forward pass through a single layer with mean-separation.

    1. Compute/use dominant direction
    2. Separate into dominant + residual
    3. Decompose residual into k subspaces
    4. Forward each (dominant + subspace_i) through next layer
    5. Average-merge results

    Args:
        model: TransformerLens model
        activations: [batch, pos, d_model] at layer_idx
        layer_idx: which layer these activations come from
        k: number of subspaces
        tokens: original tokens for hook-based forward
        dom: pre-computed dominant direction (None = compute fresh)
        recompute_dominant: if True, compute new dominant direction from these activations

    Returns:
        (merged_output, dominant_direction, stats_dict)
    """
    if dom is None or recompute_dominant:
        dom = compute_dominant_direction(activations)

    dominant_component, residual = separate_dominant(activations, dom)
    subspaces = decompose_residual_subspaces(residual, k)

    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    output_hook = f"blocks.{layer_idx + 1}.hook_resid_post"

    outputs = []
    for i, sub in enumerate(subspaces):
        # Each subspace gets the dominant component + its own residual portion
        sub_act = (dominant_component + sub).to(model.cfg.device)

        def make_hook(act):
            def hook_fn(value, hook):
                return act
            return hook_fn

        captured = {}
        def make_capture(store):
            def capture_fn(value, hook):
                store["out"] = value.detach().cpu()
                return value
            return capture_fn

        with torch.no_grad():
            model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (hook_name, make_hook(sub_act)),
                    (output_hook, make_capture(captured)),
                ]
            )
            outputs.append(captured["out"])
            torch.cuda.empty_cache()

    # Average merge
    merged = torch.stack(outputs).mean(dim=0)

    stats = {
        "dominant_var_share": dom.variance_share,
        "sv_ratio": dom.sv_ratio,
        "k": k,
        "layer": layer_idx,
    }

    return merged, dom, stats


def pda_forward_multi_layer(
    model: HookedTransformer,
    tokens: torch.Tensor,
    start_layer: int,
    n_layers: int,
    k: int,
    recompute_dominant_per_layer: bool = True,
) -> PDAResult:
    """
    Multi-layer PDA: run mean-separated parallel processing through consecutive layers.

    Args:
        model: TransformerLens model
        tokens: input tokens
        start_layer: first layer to apply PDA
        n_layers: how many consecutive layers to process with PDA
        k: number of subspaces
        recompute_dominant_per_layer: recompute dominant direction at each layer

    Returns:
        PDAResult with merged activations, comparison metrics, degradation curve
    """
    end_layer = start_layer + n_layers

    # Get activations at start_layer and ground truth at end_layer
    hooks_needed = [f"blocks.{start_layer}.hook_resid_post",
                    f"blocks.{end_layer}.hook_resid_post"]
    # Also capture intermediate layers for degradation comparison
    for l in range(start_layer + 1, end_layer):
        hooks_needed.append(f"blocks.{l}.hook_resid_post")

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hooks_needed)

    ground_truth = {l: cache[f"blocks.{l}.hook_resid_post"].detach().cpu()
                    for l in range(start_layer, end_layer + 1)}
    del cache
    torch.cuda.empty_cache()

    # Run PDA through each layer
    current_acts = ground_truth[start_layer]
    dom = None
    per_layer_cos = []

    for l in range(start_layer, end_layer):
        merged, dom, stats = pda_forward_single_layer(
            model, current_acts, l, k, tokens,
            dom=dom if not recompute_dominant_per_layer else None,
            recompute_dominant=recompute_dominant_per_layer,
        )

        # Compare merged output with ground truth at l+1
        gt = ground_truth[l + 1]
        cos = F.cosine_similarity(
            merged.reshape(-1, merged.shape[-1]),
            gt.reshape(-1, gt.shape[-1]),
            dim=1
        ).mean().item()
        per_layer_cos.append(cos)

        # Use merged as input for next layer
        current_acts = merged

    # Final comparison
    final_gt = ground_truth[end_layer]
    final_cos = F.cosine_similarity(
        current_acts.reshape(-1, current_acts.shape[-1]),
        final_gt.reshape(-1, final_gt.shape[-1]),
        dim=1
    ).mean().item()
    final_mse = torch.mean((current_acts - final_gt)**2).item()

    return PDAResult(
        merged_activations=current_acts,
        original_activations=final_gt,
        cos_sim=final_cos,
        mse=final_mse,
        n_layers=n_layers,
        k_subspaces=k,
        layer_range=(start_layer, end_layer),
        per_layer_cos_sim=per_layer_cos,
    )


def profile_all_layers(
    model: HookedTransformer,
    tokens: torch.Tensor,
    k: int = 2,
) -> Dict[int, Dict[str, float]]:
    """
    Profile every layer for PDA suitability.

    Returns dict: layer_idx -> {pr_original, pr_residual, dominant_var, cos_sim_k2}
    """
    n_layers = model.cfg.n_layers

    # Extract all activations at once
    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    all_acts = {}
    for l in range(n_layers):
        all_acts[l] = cache[f"blocks.{l}.hook_resid_post"].detach().cpu()
    del cache
    torch.cuda.empty_cache()

    results = {}
    for l in range(n_layers - 1):  # can't forward past last layer
        acts = all_acts[l]

        # Dominant direction analysis
        dom = compute_dominant_direction(acts)

        # Residual PR
        _, residual = separate_dominant(acts, dom)
        flat_r = residual.reshape(-1, residual.shape[-1]).float()
        S_r = torch.linalg.svdvals(flat_r)
        pr_r = ((S_r.sum()**2) / (S_r**2).sum()).item()

        # Original PR
        flat_o = (acts.reshape(-1, acts.shape[-1]).float() - dom.mean)
        S_o = torch.linalg.svdvals(flat_o)
        pr_o = ((S_o.sum()**2) / (S_o**2).sum()).item()

        # Quick PDA test (single layer, k=2)
        try:
            merged, _, _ = pda_forward_single_layer(model, acts, l, k, tokens, dom)
            gt = all_acts[l + 1]
            cos = F.cosine_similarity(
                merged.reshape(-1, merged.shape[-1]),
                gt.reshape(-1, gt.shape[-1]),
                dim=1
            ).mean().item()
        except Exception as e:
            cos = float('nan')

        results[l] = {
            "pr_original": pr_o,
            "pr_residual": pr_r,
            "dominant_var_share": dom.variance_share,
            "sv_ratio": dom.sv_ratio,
            "cos_sim": cos,
        }

        if l % 5 == 0:
            print(f"  Layer {l:2d}: PR {pr_o:.1f}->{pr_r:.1f}, dom={dom.variance_share:.1%}, cos={cos:.4f}")

    return results
