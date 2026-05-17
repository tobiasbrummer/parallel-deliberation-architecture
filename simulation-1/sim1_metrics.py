"""
Simulation 1: Signal Processing Metrics for PDA Convergence Diagnostics.

Toby's unique contribution -- signal processing concepts (SNR, phase coherence,
crest factor) applied as convergence diagnostics for parallel deliberation.
None of the 75+ papers in the literature review use these metrics.

Author: Toby Brummer & Claude
Date: 2026-03-29
"""

import numpy as np
from numpy.linalg import svd, norm
from typing import Optional


def compute_snr(worker_states: list[np.ndarray]) -> float:
    """
    Signal-to-Noise Ratio over worker outputs.

    Signal = consensus direction (mean of all workers).
    Noise = individual deviation from consensus.

    Returns SNR in dB. Higher = stronger consensus.
    inf = perfect agreement, negative = more noise than signal.
    """
    stacked = np.stack([s.flatten() for s in worker_states])
    mean = np.mean(stacked, axis=0)

    signal_power = norm(mean) ** 2
    noise_power = np.mean([norm(s - mean) ** 2 for s in stacked])

    if noise_power < 1e-12:
        return float("inf")
    if signal_power < 1e-12:
        return -float("inf")

    return 10.0 * np.log10(signal_power / noise_power)


def compute_phase_coherence(
    worker_states: list[np.ndarray],
    n_components: int = 5,
) -> np.ndarray:
    """
    Phase coherence per SVD component across workers.

    Decomposes the worker state matrix via SVD and measures how aligned
    workers are along each principal component. Inspired by multi-microphone
    coherence analysis in audio engineering.

    Returns array of shape (n_components,), values in [0, 1].
    1.0 = all workers agree on this component.
    0.0 = workers are split (half positive, half negative projection).
    """
    stacked = np.stack([s.flatten() for s in worker_states])

    if stacked.shape[0] < 2:
        return np.ones(min(n_components, stacked.shape[1]))

    U, S, Vt = svd(stacked, full_matrices=False)
    n_comp = min(n_components, len(S))

    coherences = np.zeros(n_comp)
    for k in range(n_comp):
        # Project all workers onto component k
        projections = stacked @ Vt[k]
        # Coherence = agreement on sign (normalized)
        # More nuanced than just sign: use normalized projections
        norms = np.abs(projections)
        if np.sum(norms) < 1e-12:
            coherences[k] = 0.0
            continue

        # Weighted sign agreement: large projections count more
        weights = norms / (np.sum(norms) + 1e-12)
        signed = np.sign(projections)
        coherences[k] = abs(np.sum(weights * signed))

    return coherences


def compute_crest_factor(worker_states: list[np.ndarray]) -> float:
    """
    Crest factor of the divergence vector between workers.

    Crest Factor = peak(|divergence|) / rms(divergence)

    High crest factor: disagreement concentrated in few dimensions
                       (specific point of contention).
    Low crest factor:  disagreement spread broadly
                       (general uncertainty).

    From audio analysis: measures whether a signal has concentrated
    peaks or is evenly distributed.
    """
    stacked = np.stack([s.flatten() for s in worker_states])
    # Per-dimension standard deviation across workers
    divergence = np.std(stacked, axis=0)

    peak = np.max(np.abs(divergence))
    rms = np.sqrt(np.mean(divergence ** 2))

    if rms < 1e-12:
        return 1.0  # No divergence at all

    return float(peak / rms)


def compute_divergence_stability(
    coherence_history: list[np.ndarray],
    window: int = 3,
) -> np.ndarray:
    """
    Measures whether incoherent components are stable over iterations.

    Stable incoherence = structured disagreement (interesting).
    Unstable incoherence = noise (ignore).

    Returns per-component stability score (0-1) over the last `window` steps.
    """
    if len(coherence_history) < window:
        return np.zeros(coherence_history[0].shape if coherence_history else 0)

    recent = np.stack(coherence_history[-window:])  # (window, n_components)
    # Stability = 1 - normalized variance over time
    variance = np.var(recent, axis=0)
    max_var = np.max(variance) if np.max(variance) > 1e-12 else 1.0
    stability = 1.0 - (variance / max_var)

    return stability


def halting_decision(
    snr: float,
    snr_prev: Optional[float],
    coherence: np.ndarray,
    crest_factor: float,
) -> str:
    """
    Multi-dimensional halting decision based on signal processing metrics.

    Returns one of:
    - "continue": keep iterating
    - "stop_confident": strong consensus, result is robust
    - "stop_uncertain": consensus with specific disagreement point
    - "stop_emergency": deliberation is destabilizing, stop immediately

    Implements the halting rules from PDA v2 specification.
    """
    # SNR trend
    if snr_prev is not None:
        snr_delta = snr - snr_prev
    else:
        snr_delta = float("inf")

    # Emergency: SNR dropping means destabilization
    if snr_delta < -3.0:  # >3dB drop is significant
        return "stop_emergency"

    # Look at top-3 components for coherence
    n_top = min(3, len(coherence))
    mean_coherence = np.mean(coherence[:n_top])

    # Strong consensus
    if snr > 20.0 and mean_coherence > 0.8:
        return "stop_confident"

    # Stagnation detection
    if snr_prev is not None and abs(snr_delta) < 0.5:
        if snr > 10.0:
            if crest_factor > 3.0:
                # Specific disagreement point, but overall ok
                return "stop_uncertain"
            else:
                # Broad stagnation at decent SNR
                return "stop_confident"

    # Moderate SNR declining slowly
    if snr_prev is not None and snr_delta < -1.0:
        return "stop_emergency"

    return "continue"


def compute_all_metrics(history: dict) -> dict:
    """
    Compute all signal processing metrics over the full iteration history.

    Args:
        history: dict with "states" key containing list of
                 list[np.ndarray] (worker states per iteration)

    Returns dict with:
        snr: list[float] - SNR per iteration
        coherence: list[np.ndarray] - phase coherence per iteration
        crest_factor: list[float] - crest factor per iteration
        halting: list[str] - halting decision per iteration
        divergence_stability: list[np.ndarray] - stability of incoherent components
    """
    metrics = {
        "snr": [],
        "coherence": [],
        "crest_factor": [],
        "halting": [],
        "divergence_stability": [],
    }

    for i, states in enumerate(history["states"]):
        snr = compute_snr(states)
        coherence = compute_phase_coherence(states)
        cf = compute_crest_factor(states)

        snr_prev = metrics["snr"][-1] if metrics["snr"] else None
        halt = halting_decision(snr, snr_prev, coherence, cf)

        stability = compute_divergence_stability(
            metrics["coherence"], window=3
        ) if metrics["coherence"] else np.zeros_like(coherence)

        metrics["snr"].append(snr)
        metrics["coherence"].append(coherence)
        metrics["crest_factor"].append(cf)
        metrics["halting"].append(halt)
        metrics["divergence_stability"].append(stability)

    return metrics


def compare_halting_methods(
    history: dict,
    signal_metrics: dict,
    delta_epsilon: float = 1e-4,
) -> dict:
    """
    Compare signal-based halting vs simple delta-threshold halting.

    Returns dict with:
        delta_stop: int - iteration where delta < epsilon
        signal_stop: int - iteration where signal metrics say stop
        signal_decision: str - what kind of stop
        delta_at_signal_stop: float - how small delta was when signal said stop
        snr_at_delta_stop: float - what SNR was when delta said stop
    """
    deltas = history.get("deltas", [])

    # Delta-based stopping
    delta_stop = len(deltas)  # default: didn't stop
    for i, d in enumerate(deltas):
        if d < delta_epsilon:
            delta_stop = i
            break

    # Signal-based stopping
    signal_stop = len(signal_metrics["halting"])
    signal_decision = "continue"
    for i, halt in enumerate(signal_metrics["halting"]):
        if halt != "continue":
            signal_stop = i
            signal_decision = halt
            break

    result = {
        "delta_stop": delta_stop,
        "signal_stop": signal_stop,
        "signal_decision": signal_decision,
    }

    # Cross-reference: what was the other metric at each stop point?
    if signal_stop < len(deltas):
        result["delta_at_signal_stop"] = deltas[signal_stop]
    if delta_stop < len(signal_metrics["snr"]):
        result["snr_at_delta_stop"] = signal_metrics["snr"][delta_stop]

    return result


if __name__ == "__main__":
    print("Testing signal processing metrics...")

    # Create some fake worker states
    np.random.seed(42)
    dim = 64

    # Case 1: Workers in strong agreement
    mean_vec = np.random.randn(dim)
    workers_agree = [mean_vec + np.random.randn(dim) * 0.1 for _ in range(5)]

    snr = compute_snr(workers_agree)
    coh = compute_phase_coherence(workers_agree)
    cf = compute_crest_factor(workers_agree)
    print(f"\nStrong agreement:")
    print(f"  SNR: {snr:.1f} dB")
    print(f"  Coherence (top 3): {coh[:3]}")
    print(f"  Crest Factor: {cf:.2f}")

    # Case 2: Workers in complete disagreement
    workers_disagree = [np.random.randn(dim) for _ in range(5)]

    snr = compute_snr(workers_disagree)
    coh = compute_phase_coherence(workers_disagree)
    cf = compute_crest_factor(workers_disagree)
    print(f"\nComplete disagreement:")
    print(f"  SNR: {snr:.1f} dB")
    print(f"  Coherence (top 3): {coh[:3]}")
    print(f"  Crest Factor: {cf:.2f}")

    # Case 3: Workers agree on most dims, disagree on a few
    workers_partial = [mean_vec.copy() for _ in range(5)]
    for w in workers_partial:
        # Flip a few dimensions randomly
        flip_dims = np.random.choice(dim, size=5, replace=False)
        w[flip_dims] += np.random.randn(5) * 5.0

    snr = compute_snr(workers_partial)
    coh = compute_phase_coherence(workers_partial)
    cf = compute_crest_factor(workers_partial)
    print(f"\nPartial disagreement (concentrated):")
    print(f"  SNR: {snr:.1f} dB")
    print(f"  Coherence (top 3): {coh[:3]}")
    print(f"  Crest Factor: {cf:.2f}")

    # Case 4: Halting decision sequence
    print(f"\nHalting decisions:")
    halt = halting_decision(5.0, None, coh, cf)
    print(f"  SNR=5, first iter: {halt}")
    halt = halting_decision(15.0, 5.0, np.array([0.9, 0.8, 0.7]), 2.0)
    print(f"  SNR=15 (rising), high coherence: {halt}")
    halt = halting_decision(25.0, 15.0, np.array([0.95, 0.9, 0.85]), 1.5)
    print(f"  SNR=25 (rising), very high coherence: {halt}")
    halt = halting_decision(10.0, 25.0, np.array([0.3, 0.2, 0.1]), 4.0)
    print(f"  SNR=10 (dropping!), low coherence: {halt}")

    print("\nAll tests passed.")
