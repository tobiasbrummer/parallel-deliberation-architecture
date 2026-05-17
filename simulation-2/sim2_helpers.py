import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import svd
from transformer_lens import HookedTransformer
from typing import List, Dict, Any, Optional

def extract_activations(model: HookedTransformer, prompts: List[str], layers: List[int]) -> Dict[int, torch.Tensor]:
    """
    Extrahiert Aktivierungen aus TransformerLens cache.
    Verarbeitet Prompts einzeln um Padding-Artefakte zu vermeiden.
    Returns: Dict[layer] -> [total_tokens, d_model] (2D, kein Padding)
    """
    all_acts = {l: [] for l in layers}

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            if hook_name in cache:
                all_acts[layer].append(cache[hook_name][0].detach().cpu())
        del cache
        torch.cuda.empty_cache()

    activations = {}
    for layer in layers:
        if all_acts[layer]:
            activations[layer] = torch.cat(all_acts[layer], dim=0)
    return activations


def extract_activations_batched(model: HookedTransformer, prompts: List[str], layers: List[int]) -> Dict[int, torch.Tensor]:
    """
    Batch-Version: Alle Prompts zusammen (mit Padding).
    Returns: Dict[layer] -> [batch, pos, d_model] (3D)
    Nötig für parallel_forward (Hook-basiert, braucht konsistente Shape).
    """
    tokens = model.to_tokens(prompts)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    activations = {}
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        if hook_name in cache:
            activations[layer] = cache[hook_name].detach().cpu()

    del cache
    torch.cuda.empty_cache()
    return activations, tokens

def compute_svd_spectrum(activations: torch.Tensor) -> Dict[str, Any]:
    """
    Singular Value Spectrum und effektive Dimensionalität (Participation Ratio).
    activations shape: [batch, pos, d_model] oder [n_samples, d_model]
    """
    if activations.ndim == 3:
        # Flatten batch and pos dimensions
        flat_acts = activations.reshape(-1, activations.shape[-1])
    else:
        flat_acts = activations
        
    # Zentrieren
    mean = flat_acts.mean(dim=0)
    centered = flat_acts - mean
    
    # SVD
    # Da wir nur S brauchen, nutzen wir torch.linalg.svdvals für Performance
    s = torch.linalg.svdvals(centered.to(torch.float32))
    s_sq = s**2
    
    # Participation Ratio: (sum S_i^2)^2 / sum (S_i^4)
    pr = (s_sq.sum()**2) / (s_sq**2).sum()
    
    # Variance explained
    var_exp = s_sq / s_sq.sum()
    cum_var_exp = torch.cumsum(var_exp, dim=0)
    
    return {
        "singular_values": s.numpy(),
        "participation_ratio": pr.item(),
        "variance_explained": var_exp.numpy(),
        "cumulative_variance_explained": cum_var_exp.numpy()
    }

def decompose_subspaces(activations: torch.Tensor, k: int) -> Dict[str, Any]:
    """
    SVD-basierte Zerlegung in k Subspaces.
    """
    if activations.ndim == 3:
        original_shape = activations.shape
        flat_acts = activations.reshape(-1, activations.shape[-1])
    else:
        original_shape = None
        flat_acts = activations
        
    mean = flat_acts.mean(dim=0)
    centered = flat_acts - mean
    
    # SVD: X = U S V^T
    U, S, Vh = torch.linalg.svd(centered.to(torch.float32), full_matrices=False)
    V = Vh.T # V contains the basis vectors in columns
    
    # Zerlegung in k gleich große (oder fast gleich große) Teile der S-Werte
    # Alternative: Top k Komponenten vs Rest. 
    # Hier: Wir nehmen die Top n_comp und teilen sie in k Subspaces auf?
    # Der User-Plan sagt "decompose_subspaces(activations, k)".
    # Wir teilen die d_model Dimensionen in k gleich große orthogonale Blöcke der SVD-Basis auf.
    
    d_model = flat_acts.shape[-1]
    block_size = d_model // k
    
    subspaces = []
    for i in range(k):
        start = i * block_size
        end = (i + 1) * block_size if i < k - 1 else d_model
        
        # Komponenten in diesem Subspace
        # Wir speichern die Koeffizienten (U * S) für diesen Block
        comp_k = U[:, start:end] @ torch.diag(S[start:end])
        basis_k = V[:, start:end]
        subspaces.append({
            "components": comp_k,
            "basis": basis_k,
            "indices": (start, end)
        })
        
    return {
        "subspaces": subspaces,
        "mean": mean,
        "original_shape": original_shape
    }

def reconstruct_from_subspaces(subspaces: List[Dict], mean: torch.Tensor, original_shape: Optional[tuple] = None) -> torch.Tensor:
    """
    Rekonstruktion aus den Teil-Subspaces.
    """
    reconstructed_flat = torch.zeros((subspaces[0]["components"].shape[0], mean.shape[0]))
    
    for sub in subspaces:
        reconstructed_flat += sub["components"] @ sub["basis"].T
        
    reconstructed_flat += mean
    
    if original_shape:
        return reconstructed_flat.reshape(original_shape)
    return reconstructed_flat

def parallel_forward(model: HookedTransformer, activations: torch.Tensor, layer_idx: int, k_subspaces: int, tokens: torch.Tensor = None) -> List[torch.Tensor]:
    """
    Einzelne Subspaces getrennt durch nächsten Layer schicken.
    activations: hook_resid_post von layer_idx, shape [batch, pos, d_model]
    tokens: Original-Tokens (nötig für den Forward Pass)
    """
    decomp = decompose_subspaces(activations, k_subspaces)
    mean = decomp["mean"]
    original_shape = decomp["original_shape"]

    outputs = []
    output_hook_name = f"blocks.{layer_idx+1}.hook_resid_post"

    for i in range(k_subspaces):
        sub = decomp["subspaces"][i]
        sub_act_flat = (sub["components"] @ sub["basis"].T) + mean
        sub_act = sub_act_flat.reshape(original_shape).to(model.cfg.device)

        # Closure muss sub_act per Iteration binden
        def make_hook(act):
            def hook_fn(value, hook):
                return act
            return hook_fn

        with torch.no_grad():
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            captured = {}
            def make_capture(store):
                def capture_fn(value, hook):
                    store["out"] = value.detach().cpu()
                    return value
                return capture_fn

            model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (hook_name, make_hook(sub_act)),
                    (output_hook_name, make_capture(captured))
                ]
            )
            outputs.append(captured["out"])
            torch.cuda.empty_cache()

    return outputs

def merge_subspace_outputs(outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Average-Merge der parallelen Outputs.
    """
    return torch.stack(outputs).mean(dim=0)

def compute_kl_divergence(logits_original: torch.Tensor, logits_reconstructed: torch.Tensor) -> float:
    """
    KL-Divergenz zwischen Original und Rekonstruktion.
    logits: [batch, pos, d_vocab]
    """
    p = F.softmax(logits_original, dim=-1)
    log_p = F.log_softmax(logits_original, dim=-1)
    log_q = F.log_softmax(logits_reconstructed, dim=-1)
    
    # KL(P || Q) = sum P * (log P - log Q)
    kl = torch.sum(p * (log_p - log_q), dim=-1).mean()
    return kl.item()

def compute_token_accuracy(logits_original: torch.Tensor, logits_reconstructed: torch.Tensor) -> float:
    """
    Top-1 Match zwischen Original und Rekonstruktion.
    """
    pred_orig = torch.argmax(logits_original, dim=-1)
    pred_recon = torch.argmax(logits_reconstructed, dim=-1)
    
    correct = (pred_orig == pred_recon).float().mean()
    return correct.item()
