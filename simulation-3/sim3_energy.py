"""
Simulation 3: Energy Minimization on Real Activations

Adapts the energy minimization approach from Sim 1 to work on
real transformer activations with mean-separated subspaces.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulation-1"))
from sim1_metrics import (
    compute_snr as sim1_snr,
    compute_phase_coherence as sim1_coherence,
    compute_crest_factor as sim1_crest,
)


def compute_energy(subspace_outputs: List[torch.Tensor]) -> float:
    """
    Energy function: sum of pairwise distances between subspace outputs.
    Lower energy = more agreement between subspaces.

    Args:
        subspace_outputs: list of k tensors, each [batch, pos, d_model]
    """
    k = len(subspace_outputs)
    energy = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            diff = subspace_outputs[i] - subspace_outputs[j]
            energy += torch.mean(diff**2).item()
    return energy / (k * (k - 1) / 2)  # normalize by number of pairs


def energy_gradient_step(
    subspace_outputs: List[torch.Tensor],
    lr: float = 0.1,
) -> List[torch.Tensor]:
    """
    One gradient step of energy minimization.
    Moves each subspace output towards the mean of all others.

    This is equivalent to gradient descent on the pairwise distance energy,
    which has the closed-form update: x_i <- x_i + lr * (mean_others - x_i)
    """
    k = len(subspace_outputs)
    total = torch.stack(subspace_outputs).sum(dim=0)  # [batch, pos, d_model]

    updated = []
    for i in range(k):
        # Mean of all others
        mean_others = (total - subspace_outputs[i]) / (k - 1)
        # Gradient step towards mean of others
        new_xi = subspace_outputs[i] + lr * (mean_others - subspace_outputs[i])
        updated.append(new_xi)

    return updated


def compute_signal_metrics(subspace_outputs: List[torch.Tensor]) -> Dict[str, float]:
    """
    Compute signal-processing metrics on subspace outputs.
    Adapts sim1 metrics for real activations.
    """
    # Convert to flat numpy arrays for sim1 metrics
    flat_arrays = []
    for out in subspace_outputs:
        flat_arrays.append(out.reshape(-1).float().numpy())

    snr = sim1_snr(flat_arrays)
    coherence = sim1_coherence(flat_arrays, n_components=5)
    crest = sim1_crest(flat_arrays)

    return {
        "snr": snr,
        "phase_coherence_mean": float(np.mean(coherence)),
        "phase_coherence": coherence.tolist() if isinstance(coherence, np.ndarray) else coherence,
        "crest_factor": crest,
    }


def iterate_energy(
    subspace_outputs: List[torch.Tensor],
    lr: float = 0.1,
    max_iter: int = 50,
    halt_on_snr: bool = True,
    snr_threshold: float = 10.0,
    energy_threshold: float = 1e-4,
    verbose: bool = False,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """
    Run energy minimization with signal-metric-based halting.

    From Sim 1: Signal metrics (SNR) are better halting criteria than delta.
    "Convergence in delta-space does not imply convergence in signal quality."

    Args:
        subspace_outputs: list of k tensors [batch, pos, d_model]
        lr: learning rate (0.1 optimal from Sim 1)
        max_iter: maximum iterations
        halt_on_snr: use SNR for halting (True) or energy delta (False)
        snr_threshold: halt when SNR exceeds this (dB)
        energy_threshold: halt when energy change < this (fallback)
        verbose: print progress

    Returns:
        (refined_outputs, history_dict)
    """
    history = {
        "energy": [],
        "snr": [],
        "coherence": [],
        "crest": [],
        "iterations": 0,
        "halted_by": "max_iter",
    }

    current = [o.clone() for o in subspace_outputs]
    prev_energy = float('inf')

    for iteration in range(max_iter):
        energy = compute_energy(current)
        history["energy"].append(energy)

        # Signal metrics (compute every iteration for tracking)
        signals = compute_signal_metrics(current)
        history["snr"].append(signals["snr"])
        history["coherence"].append(signals["phase_coherence_mean"])
        history["crest"].append(signals["crest_factor"])

        if verbose and iteration % 5 == 0:
            print(f"  Iter {iteration:3d}: E={energy:.6f}, SNR={signals['snr']:.2f}dB, "
                  f"Coh={signals['phase_coherence_mean']:.4f}")

        # Halting criteria
        if halt_on_snr and signals["snr"] >= snr_threshold:
            history["halted_by"] = f"snr>={snr_threshold}"
            history["iterations"] = iteration + 1
            break

        energy_delta = abs(prev_energy - energy)
        if energy_delta < energy_threshold and iteration > 0:
            history["halted_by"] = f"energy_delta<{energy_threshold}"
            history["iterations"] = iteration + 1
            break

        prev_energy = energy

        # Gradient step
        current = energy_gradient_step(current, lr=lr)

    else:
        history["iterations"] = max_iter

    return current, history


def merge_with_energy(
    subspace_outputs: List[torch.Tensor],
    lr: float = 0.1,
    max_iter: int = 30,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Convenience function: run energy minimization then average-merge.

    Returns:
        (merged_tensor, history)
    """
    refined, history = iterate_energy(
        subspace_outputs, lr=lr, max_iter=max_iter, verbose=verbose
    )
    merged = torch.stack(refined).mean(dim=0)
    return merged, history
