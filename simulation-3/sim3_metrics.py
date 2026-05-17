"""
Simulation 3: Output Quality Metrics

Goes beyond activation-level metrics (CosSim, MSE) to actual text output quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any
from transformer_lens import HookedTransformer


def compute_topk_overlap(logits_orig: torch.Tensor, logits_pda: torch.Tensor, k: int = 10) -> Dict[str, float]:
    """
    How many of the top-k predicted tokens overlap between original and PDA?

    Args:
        logits_orig: [batch, pos, vocab_size]
        logits_pda: [batch, pos, vocab_size]
        k: number of top tokens to compare

    Returns:
        dict with mean_overlap (fraction), top1_match (fraction), per_position stats
    """
    topk_orig = torch.topk(logits_orig, k, dim=-1).indices  # [batch, pos, k]
    topk_pda = torch.topk(logits_pda, k, dim=-1).indices

    # Top-1 match
    top1_match = (topk_orig[..., 0] == topk_pda[..., 0]).float().mean().item()

    # Top-k overlap per position
    overlaps = []
    for b in range(logits_orig.shape[0]):
        for p in range(logits_orig.shape[1]):
            orig_set = set(topk_orig[b, p].tolist())
            pda_set = set(topk_pda[b, p].tolist())
            overlap = len(orig_set & pda_set) / k
            overlaps.append(overlap)

    return {
        "topk_overlap_mean": np.mean(overlaps),
        "topk_overlap_std": np.std(overlaps),
        "top1_match": top1_match,
        "k": k,
    }


def compute_kl_divergence(logits_orig: torch.Tensor, logits_pda: torch.Tensor) -> float:
    """KL(original || pda) averaged over all positions."""
    p = F.softmax(logits_orig, dim=-1)
    log_p = F.log_softmax(logits_orig, dim=-1)
    log_q = F.log_softmax(logits_pda, dim=-1)

    kl = torch.sum(p * (log_p - log_q), dim=-1).mean()
    return kl.item()


def compute_perplexity_from_logits(logits: torch.Tensor, target_tokens: torch.Tensor) -> float:
    """
    Perplexity of logits against target tokens.
    Lower = model is more confident about the right tokens.

    Args:
        logits: [batch, pos, vocab_size] - shifted by 1 (logits predict next token)
        target_tokens: [batch, pos] - the actual tokens
    """
    # Shift: logits at position i predict token at position i+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = target_tokens[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_targets.reshape(-1),
        reduction='mean'
    )
    return torch.exp(loss).item()


def generate_and_compare(
    model: HookedTransformer,
    prompt: str,
    n_tokens: int = 30,
    pda_hook_fn=None,
) -> Dict[str, any]:
    """
    Generate tokens with and without PDA, compare results.

    Args:
        model: TransformerLens model
        prompt: input text
        n_tokens: tokens to generate
        pda_hook_fn: if provided, called with (tokens, model) to get PDA logits
                     for each step. If None, only generates normal output.

    Returns:
        dict with original_text, original_tokens, and optionally pda comparison
    """
    tokens = model.to_tokens(prompt)

    # Normal generation
    normal_tokens = tokens.clone()
    for _ in range(n_tokens):
        with torch.no_grad():
            logits = model(normal_tokens)
        next_token = logits[0, -1].argmax()
        normal_tokens = torch.cat([normal_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    normal_text = model.to_string(normal_tokens[0])

    result = {
        "prompt": prompt,
        "normal_text": normal_text,
        "normal_tokens": normal_tokens[0].tolist(),
        "n_generated": n_tokens,
    }

    return result


def compute_output_quality_suite(
    model: HookedTransformer,
    tokens: torch.Tensor,
    logits_orig: torch.Tensor,
    logits_pda: torch.Tensor,
) -> Dict[str, float]:
    """
    Full quality comparison between original and PDA logits.

    Returns dict with all metrics.
    """
    kl = compute_kl_divergence(logits_orig, logits_pda)
    topk = compute_topk_overlap(logits_orig, logits_pda, k=10)
    topk5 = compute_topk_overlap(logits_orig, logits_pda, k=5)

    ppl_orig = compute_perplexity_from_logits(logits_orig, tokens)
    ppl_pda = compute_perplexity_from_logits(logits_pda, tokens)

    return {
        "kl_divergence": kl,
        "top1_match": topk["top1_match"],
        "top5_overlap": topk5["topk_overlap_mean"],
        "top10_overlap": topk["topk_overlap_mean"],
        "perplexity_original": ppl_orig,
        "perplexity_pda": ppl_pda,
        "perplexity_ratio": ppl_pda / ppl_orig if ppl_orig > 0 else float('inf'),
    }
