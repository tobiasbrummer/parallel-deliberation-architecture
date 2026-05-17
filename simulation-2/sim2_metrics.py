import numpy as np
import torch
import sys
import os

# Pfad anpassen um sim1_metrics zu importieren
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation-1")))
from sim1_metrics import compute_snr as sim1_snr, compute_phase_coherence as sim1_coherence, compute_crest_factor as sim1_crest

def adapt_activations(activations):
    """Konvertiert torch.Tensor zu numpy-Liste für sim1-Metriken."""
    if isinstance(activations, torch.Tensor):
        # Wenn es ein Batch ist, behandeln wir die Samples als 'Worker'
        if activations.ndim == 3:
            # [batch, pos, d_model] -> Liste von [pos * d_model]
            return [act.flatten().numpy() for act in activations]
        else:
            # [n_samples, d_model]
            return [act.numpy() for act in activations]
    return activations

def compute_snr(activations):
    """Adaptiertes SNR für echte Aktivierungen."""
    return sim1_snr(adapt_activations(activations))

def compute_phase_coherence(activations, n_components=5):
    """Adaptierte Phasenkohärenz für echte Aktivierungen."""
    return sim1_coherence(adapt_activations(activations), n_components)

def compute_crest_factor(activations):
    """Adaptierter Crest Factor für echte Aktivierungen."""
    return sim1_crest(adapt_activations(activations))

def correlation_analysis(signal_metrics: dict, quality_metrics: dict):
    """
    Korrelation zwischen Signalmetriken (SNR, Coherence, Crest) 
    und Output-Qualität (KL-Div, Accuracy).
    signal_metrics: {'snr': [...], 'coherence': [...], 'crest': [...]}
    quality_metrics: {'kl': [...], 'acc': [...]}
    """
    results = {}
    
    for s_name, s_vals in signal_metrics.items():
        s_vals = np.array(s_vals)
        if s_vals.ndim > 1:
            # Bei Kohärenz nehmen wir den Durchschnitt der Top-Komponenten
            s_vals = np.mean(s_vals, axis=1)
            
        for q_name, q_vals in quality_metrics.items():
            q_vals = np.array(q_vals)
            
            # Korrelation berechnen
            if len(s_vals) == len(q_vals):
                corr = np.corrcoef(s_vals, q_vals)[0, 1]
                results[f"{s_name}_vs_{q_name}"] = corr
                
    return results
