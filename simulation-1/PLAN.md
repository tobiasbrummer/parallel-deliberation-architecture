# Simulation 1: Mathematische Konvergenz -- Detailplan

Status: Bereit zur Implementierung
Datum: 2026-03-29
Autoren: Toby, Claude
Abhaengigkeiten: numpy, torch, matplotlib, seaborn, scipy
GPU: Nicht noetig (reine Mathematik, CPU reicht)

## Ziel

Klaeren ob iterative parallele Verarbeitung in orthogonalen Unterraeumen
mathematisch konvergiert -- BEVOR ein neuronales Netz involviert wird.
Gleichzeitig: Tobys Signalverarbeitungs-Metriken als Konvergenz-Diagnostik
validieren (der unique Beitrag, der in keinem der 75+ recherchierten Papers auftaucht).

## Notebook-Struktur

```
simulation-1-convergence.ipynb

  0. Setup & Imports
  1. Hilfsfunktionen
     1a. Unterraum-Generierung (orthogonal, fast-orthogonal, nicht-orthogonal)
     1b. Semantische Vektoren (Embedding-basiert)
     1c. Merge-Strategien
     1d. Konvergenz-Mechanismen (Fixpunkt, Energie)
     1e. Signalverarbeitungs-Metriken (SNR, Phasenkohaerenz, Crest Factor)
     1f. Diversity-Enforcement
  2. Experiment A: Basis-Konvergenz (Fixpunkt vs. Energie)
  3. Experiment B: Orthogonalitaetsgrad-Sweep
  4. Experiment C: Merge-Strategien-Vergleich
  5. Experiment D: Signalmetriken als Diagnostik
  6. Experiment E: Diversity-Enforcement
  7. Experiment F: Semantische vs. zufaellige Vektoren
  8. Zusammenfassung & Entscheidung
```

## 0. Setup

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.linalg import svd, orth
from dataclasses import dataclass
from typing import Literal
import json

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
```

## 1. Hilfsfunktionen

### 1a. Unterraum-Generierung

```python
def generate_orthogonal_subspaces(
    n_subspaces: int,
    dim: int,
    subspace_dim: int,
    orthogonality: float = 1.0  # 1.0 = exakt, 0.0 = zufaellig
) -> list[np.ndarray]:
    """
    Erzeugt n_subspaces Projektionsmatrizen (dim x subspace_dim).
    orthogonality steuert wie orthogonal die Unterraeume zueinander sind.

    Bei orthogonality=1.0: Unterraeume sind exakt orthogonal (Gram-Schmidt).
    Bei orthogonality<1.0: Interpolation zwischen orthogonal und zufaellig.
    """
    # Zufaellige orthogonale Basis des Gesamtraums
    full_basis = orth(np.random.randn(dim, dim))

    if n_subspaces * subspace_dim > dim:
        raise ValueError(
            f"Kann {n_subspaces} orthogonale Unterraeume der Dimension "
            f"{subspace_dim} nicht in Dimension {dim} unterbringen"
        )

    subspaces = []
    for i in range(n_subspaces):
        # Exakt orthogonaler Unterraum
        start = i * subspace_dim
        ortho_basis = full_basis[:, start:start + subspace_dim]

        if orthogonality < 1.0:
            # Zufaelliger Unterraum
            random_basis = orth(np.random.randn(dim, subspace_dim))
            # Interpolation (SLERP-artig auf Grassmann-Mannigfaltigkeit
            # -- vereinfacht als lineare Interpolation + Re-Orthogonalisierung)
            mixed = orthogonality * ortho_basis + (1 - orthogonality) * random_basis
            mixed = orth(mixed)
            subspaces.append(mixed)
        else:
            subspaces.append(ortho_basis)

    return subspaces
```

### 1b. Semantische Vektoren

```python
def load_semantic_vectors(n_vectors: int, dim: int) -> np.ndarray:
    """
    Laedt vortrainierte Embeddings und projiziert sie auf dim Dimensionen.
    Fallback: Strukturierte (nicht zufaellige) Vektoren mit Cluster-Struktur.
    """
    try:
        # Versuch: GloVe oder word2vec laden
        # (gensim oder lokale Datei)
        raise ImportError("Placeholder -- implementieren wenn verfuegbar")
    except ImportError:
        # Fallback: Cluster-strukturierte Vektoren
        # Simuliert semantische Struktur: Gruppen von aehnlichen Vektoren
        # mit klaren Zwischen-Cluster-Abstaenden
        n_clusters = max(3, n_vectors // 10)
        centers = np.random.randn(n_clusters, dim)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

        vectors = []
        for i in range(n_vectors):
            cluster = i % n_clusters
            noise = np.random.randn(dim) * 0.3
            vec = centers[cluster] + noise
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)

        return np.array(vectors)
```

### 1c. Merge-Strategien

```python
def merge_average(worker_states: list[np.ndarray]) -> np.ndarray:
    """Gewichteter Durchschnitt (Baseline)."""
    return np.mean(worker_states, axis=0)

def merge_phase_alignment(worker_states: list[np.ndarray]) -> np.ndarray:
    """
    Phase-Alignment: Phasengleiche Anteile verstaerken,
    gegenlaeufige Anteile daempfen.
    """
    mean = np.mean(worker_states, axis=0)
    aligned = np.zeros_like(mean)
    total_weight = 0

    for state in worker_states:
        # Cosine Similarity zum Durchschnitt als "Phase"
        cos_sim = np.dot(state.flatten(), mean.flatten()) / (
            np.linalg.norm(state) * np.linalg.norm(mean) + 1e-8
        )
        # Gewichtung: phasengleich = hoher Beitrag, gegenlaeufig = niedrig
        weight = max(0, cos_sim)  # Nur konstruktive Interferenz
        aligned += weight * state
        total_weight += weight

    return aligned / (total_weight + 1e-8)

def merge_frequency_selective(worker_states: list[np.ndarray]) -> np.ndarray:
    """
    Frequenz-selektiv: SVD-Zerlegung, dominante Komponenten mitteln,
    feine Komponenten staerker gewichten.
    """
    stacked = np.stack(worker_states)  # (n_workers, dim)
    U, S, Vt = svd(stacked, full_matrices=False)

    # Dominante Komponenten (niedrige "Frequenz"): einfach mitteln
    # Feine Komponenten (hohe "Frequenz"): staerker gewichten
    n_dominant = max(1, len(S) // 2)

    # Rekonstruktion mit frequenzabhaengiger Gewichtung
    result = np.zeros(worker_states[0].shape)
    for i, (s, v) in enumerate(zip(S, Vt)):
        weight = 1.0 if i < n_dominant else 2.0  # Feine Details staerker
        contribution = s * v * weight
        result += contribution / len(S)

    return result

def merge_sidechain(worker_states: list[np.ndarray]) -> np.ndarray:
    """
    Sidechain: Erster Worker liefert "Was", zweiter moduliert "Wie stark".
    Bei >2 Workern: Erster vs. Durchschnitt der restlichen.
    """
    if len(worker_states) < 2:
        return worker_states[0]

    primary = worker_states[0]
    modulator = np.mean(worker_states[1:], axis=0)

    # Elementweise Modulation (Sigmoid des Modulators als Gating)
    gate = 1.0 / (1.0 + np.exp(-modulator))
    return primary * gate

MERGE_STRATEGIES = {
    "average": merge_average,
    "phase_alignment": merge_phase_alignment,
    "frequency_selective": merge_frequency_selective,
    "sidechain": merge_sidechain,
}
```

### 1d. Konvergenz-Mechanismen

```python
def iterate_fixpoint(
    worker_states: list[np.ndarray],
    subspaces: list[np.ndarray],
    merge_fn,
    max_iter: int = 50,
    epsilon: float = 1e-6,
    diversity_strength: float = 0.0,
) -> dict:
    """
    Fixpunkt-Iteration: Merge -> Re-Projektion -> Repeat.
    Gibt History aller Zwischenschritte zurueck.
    """
    history = {"states": [], "deltas": [], "merged": []}
    states = [s.copy() for s in worker_states]

    for t in range(max_iter):
        history["states"].append([s.copy() for s in states])

        # Merge
        merged = merge_fn(states)
        history["merged"].append(merged.copy())

        # Re-Projektion in Unterraeume + Perturbation aus Merge
        new_states = []
        for i, (state, P) in enumerate(zip(states, subspaces)):
            # Projektion des Merge-Ergebnisses in den eigenen Unterraum
            projected = P @ (P.T @ merged.flatten())
            # Mischung: eigener Zustand + projiziertes Merge-Ergebnis
            new_state = 0.7 * state.flatten() + 0.3 * projected
            new_states.append(new_state)

        # Diversity-Enforcement (kontrastive Repulsion)
        if diversity_strength > 0:
            new_states = apply_diversity_repulsion(
                new_states, subspaces, diversity_strength
            )

        # Delta messen
        deltas = [np.linalg.norm(n - o) for n, o in zip(new_states, states)]
        max_delta = max(deltas)
        history["deltas"].append(max_delta)

        states = [s.copy() for s in new_states]

        if max_delta < epsilon:
            break

    history["states"].append([s.copy() for s in states])
    history["n_iterations"] = t + 1
    history["converged"] = max_delta < epsilon
    return history

def iterate_energy(
    worker_states: list[np.ndarray],
    subspaces: list[np.ndarray],
    merge_fn,
    max_iter: int = 50,
    epsilon: float = 1e-6,
    lr: float = 0.1,
    diversity_strength: float = 0.0,
) -> dict:
    """
    Energieminimierung (EBT-inspiriert):
    E(x) = Summe paarweiser Distanzen + Orthogonalitaets-Regularisierung.
    Gradient Descent auf Worker-Zustaende.
    """
    history = {"states": [], "deltas": [], "energies": [], "merged": []}
    # Torch fuer Autograd
    states_t = [torch.tensor(s, dtype=torch.float32, requires_grad=True)
                for s in worker_states]
    subspaces_t = [torch.tensor(P, dtype=torch.float32) for P in subspaces]

    for t in range(max_iter):
        history["states"].append([s.detach().numpy().copy() for s in states_t])

        # Energie berechnen
        energy = torch.tensor(0.0)

        # Paarweise Distanzen (Konsensus-Kraft)
        for i in range(len(states_t)):
            for j in range(i + 1, len(states_t)):
                dist = torch.norm(states_t[i] - states_t[j])
                energy = energy + dist ** 2

        # Diversity-Repulsion (verhindert Kollaps)
        if diversity_strength > 0:
            for i in range(len(states_t)):
                for j in range(i + 1, len(states_t)):
                    cos_sim = torch.nn.functional.cosine_similarity(
                        states_t[i].unsqueeze(0), states_t[j].unsqueeze(0)
                    )
                    energy = energy - diversity_strength * (1 - cos_sim)

        history["energies"].append(energy.item())

        # Gradient Descent
        energy.backward()

        new_states = []
        for i, (s, P) in enumerate(zip(states_t, subspaces_t)):
            with torch.no_grad():
                grad = s.grad if s.grad is not None else torch.zeros_like(s)
                new_s = s - lr * grad
                # Zurueck-Projektion in den eigenen Unterraum
                projected = P @ (P.T @ new_s.flatten())
                new_states.append(projected)

            s.grad = None

        # Delta messen
        deltas = [torch.norm(n - o).item()
                  for n, o in zip(new_states, states_t)]
        max_delta = max(deltas)
        history["deltas"].append(max_delta)

        # Merge fuer Diagnostik
        merged = merge_fn([s.detach().numpy() for s in new_states])
        history["merged"].append(merged.copy())

        states_t = [s.clone().detach().requires_grad_(True) for s in new_states]

        if max_delta < epsilon:
            break

    history["states"].append([s.detach().numpy().copy() for s in states_t])
    history["n_iterations"] = t + 1
    history["converged"] = max_delta < epsilon
    return history
```

### 1e. Signalverarbeitungs-Metriken

```python
def compute_snr(worker_states: list[np.ndarray]) -> float:
    """
    SNR: ||mean(workers)||^2 / mean(||worker_i - mean||^2)
    Signal = Konsensus-Richtung. Noise = individuelle Abweichung.
    """
    stacked = np.stack([s.flatten() for s in worker_states])
    mean = np.mean(stacked, axis=0)
    signal_power = np.linalg.norm(mean) ** 2
    noise_power = np.mean([np.linalg.norm(s - mean) ** 2 for s in stacked])
    if noise_power < 1e-12:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)  # in dB

def compute_phase_coherence(worker_states: list[np.ndarray], n_components: int = 5) -> np.ndarray:
    """
    Phasenkohaerenz pro SVD-Komponente.
    Gibt Array der Laenge n_components zurueck, Werte 0-1.
    """
    stacked = np.stack([s.flatten() for s in worker_states])
    U, S, Vt = svd(stacked, full_matrices=False)
    n_comp = min(n_components, len(S))

    coherences = []
    for k in range(n_comp):
        # Projektion aller Worker auf Komponente k
        projections = stacked @ Vt[k]
        # Kohaerenz = Varianz der Vorzeichen / mittlere Amplitude
        signs = np.sign(projections)
        coherence = abs(np.mean(signs))  # 1.0 = alle gleich, 0.0 = gemischt
        coherences.append(coherence)

    return np.array(coherences)

def compute_crest_factor(worker_states: list[np.ndarray]) -> float:
    """
    Crest Factor der Divergenz: max(|divergenz|) / rms(divergenz).
    Hoch = konzentrierter Streitpunkt. Niedrig = diffuse Unsicherheit.
    """
    stacked = np.stack([s.flatten() for s in worker_states])
    mean = np.mean(stacked, axis=0)
    divergence = np.std(stacked, axis=0)  # Pro Dimension

    peak = np.max(np.abs(divergence))
    rms = np.sqrt(np.mean(divergence ** 2))
    if rms < 1e-12:
        return 1.0
    return peak / rms

def compute_all_metrics(history: dict) -> dict:
    """Berechnet alle Metriken ueber die gesamte Iterationsgeschichte."""
    metrics = {"snr": [], "coherence": [], "crest_factor": []}
    for states in history["states"]:
        metrics["snr"].append(compute_snr(states))
        metrics["coherence"].append(compute_phase_coherence(states))
        metrics["crest_factor"].append(compute_crest_factor(states))
    return metrics
```

### 1f. Diversity-Enforcement

```python
def apply_diversity_repulsion(
    states: list[np.ndarray],
    subspaces: list[np.ndarray],
    strength: float = 0.1,
) -> list[np.ndarray]:
    """
    Kontrastive Repulsion: Verhindert Worker-Kollaps.
    Stosst Worker auseinander, die zu aehnlich werden.
    Inspiriert von PLR (Tang et al.) und Coda-Forno et al.
    """
    new_states = [s.copy() for s in states]
    for i in range(len(states)):
        repulsion = np.zeros_like(states[i])
        for j in range(len(states)):
            if i == j:
                continue
            diff = states[i].flatten() - states[j].flatten()
            dist = np.linalg.norm(diff) + 1e-8
            # Repulsive Kraft, umgekehrt proportional zur Distanz
            repulsion += strength * diff / (dist ** 2)
        new_states[i] = states[i].flatten() + repulsion
    return new_states
```

## 2. Experiment A: Basis-Konvergenz

Kernfrage: Konvergiert das System ueberhaupt?

```python
# Parameter
configs = {
    "n_subspaces": [2, 3, 5, 10],
    "dim": [64, 256, 1024],
    "mechanism": ["fixpoint", "energy"],
}

# Fuer jede Konfiguration:
# - Exakt orthogonale Unterraeume
# - Zufaellige Startvektoren in den Unterraeumen
# - merge_average als Baseline-Merge
# - Messen: Konvergenz ja/nein, Iterationen, Endstabiliaet

results_a = []
for n_sub in configs["n_subspaces"]:
    for dim in configs["dim"]:
        subspace_dim = dim // (n_sub * 2)  # Konservativ
        subspaces = generate_orthogonal_subspaces(n_sub, dim, subspace_dim)

        # Zufaellige Startzustaende in den Unterraeumen
        workers = [P @ np.random.randn(subspace_dim) for P in subspaces]

        for mechanism in configs["mechanism"]:
            if mechanism == "fixpoint":
                history = iterate_fixpoint(workers, subspaces, merge_average)
            else:
                history = iterate_energy(workers, subspaces, merge_average)

            metrics = compute_all_metrics(history)
            results_a.append({
                "n_subspaces": n_sub,
                "dim": dim,
                "mechanism": mechanism,
                "converged": history["converged"],
                "iterations": history["n_iterations"],
                "final_snr": metrics["snr"][-1],
                "snr_curve": metrics["snr"],
            })

# Visualisierung: Heatmap (n_subspaces x dim), farbkodiert nach Iterationen
# Getrennt fuer Fixpunkt und Energie
```

### Erwartetes Ergebnis
- Beides konvergiert bei exakter Orthogonalitaet
- Energieminimierung: Glattere Konvergenz, weniger sensitiv gegenueber n_subspaces
- Fixpunkt: Schneller bei wenigen Unterraeumen, instabiler bei vielen

## 3. Experiment B: Orthogonalitaetsgrad-Sweep

Kernfrage: Ab welchem Winkel bricht die Konvergenz?

```python
orthogonality_levels = [1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]

# Fuer dim=256, n_subspaces=5, beide Mechanismen:
# Sweep ueber orthogonality_levels
# Messen: Konvergenz ja/nein, Iterationen, finale Qualitaet

# Plot: X = Orthogonalitaetsgrad, Y = Iterationen bis Konvergenz
# Zwei Linien: Fixpunkt vs. Energie
# Markierung wo Konvergenz zusammenbricht
```

### Erwartetes Ergebnis
- Energieminimierung toleranter gegenueber Nicht-Orthogonalitaet
- Bruchpunkt bei Fixpunkt: wahrscheinlich um 0.7-0.8
- Signal-Merges (Phase-Alignment) verschieben den Bruchpunkt nach unten

## 4. Experiment C: Merge-Strategien-Vergleich

Kernfrage: Welcher Merge erhaelt am meisten Information?

```python
# Fuer dim=256, n_subspaces=5, Fixpunkt + Energie:
# Alle vier Merge-Strategien
# Bei drei Orthogonalitaetsgraden: 1.0, 0.8, 0.5

# Metriken pro Merge:
# - Konvergenzgeschwindigkeit
# - Finale SNR
# - Stabilitaet (Perturbationstest: leicht stoeren, zurueck?)

# Perturbationstest:
def test_stability(history, subspaces, merge_fn, perturbation_scale=0.01):
    """Stoert den Endzustand leicht und prueft ob er zurueckkonvergiert."""
    final_states = history["states"][-1]
    perturbed = [s + np.random.randn(*s.shape) * perturbation_scale
                 for s in final_states]
    recovery = iterate_fixpoint(perturbed, subspaces, merge_fn, max_iter=20)
    return recovery["converged"]
```

### Erwartetes Ergebnis
- Phase-Alignment: Robuster als Average bei Nicht-Orthogonalitaet
- Frequency-Selective: Langsamere Konvergenz, aber hoehere finale Qualitaet
- Sidechain: Asymmetrisch, gut bei klarer Worker-Hierarchie

## 5. Experiment D: Signalmetriken als Diagnostik

DER zentrale Test fuer Tobys Beitrag.

```python
# Fuer die besten Konfigurationen aus A-C:
# Alle drei Signalmetriken ueber die gesamte Iterationsgeschichte berechnen

# Analyse:
# 1. Korrelation SNR-Kurve <-> Konvergenz-Guete
# 2. Phasenkohaerenz: Identifiziert sie die "interessanten" Dimensionen?
# 3. Crest Factor: Unterscheidet er konzentrierte von diffuser Uneinigkeit?
# 4. Kombinations-Halting: Implementiere die Halting-Regeln aus PDA v2
#    und vergleiche sie mit einfachem Delta-Threshold

def halting_decision(snr, snr_prev, coherence, crest_factor):
    """
    Mehrdimensionale Halting-Entscheidung nach PDA-Spezifikation.
    Gibt zurueck: "continue", "stop_confident", "stop_uncertain", "stop_emergency"
    """
    snr_delta = snr - snr_prev if snr_prev is not None else float("inf")

    if snr_delta < -1.0:  # SNR sinkt: Destabilisierung
        return "stop_emergency"

    mean_coherence = np.mean(coherence[:3])  # Top-3 Komponenten

    if snr > 20 and mean_coherence > 0.8:
        return "stop_confident"

    if snr > 10 and abs(snr_delta) < 0.5:  # Stagnation
        if crest_factor > 3.0:
            return "stop_uncertain"  # Spezifischer Streitpunkt
        else:
            return "stop_confident"  # Diffuse, aber stabile Loesung

    return "continue"

# Vergleich: Halting via Signal-Metriken vs. einfaches Delta < epsilon
# Messen: Wann stoppt jede Methode? Wie gut ist das Ergebnis zum Stoppzeitpunkt?
```

### Erwartetes Ergebnis
- SNR trackt Konvergenzqualitaet besser als reines Delta (mehrdimensional)
- Phasenkohaerenz identifiziert Dimensionen wo Konsensus existiert vs. nicht
- Crest Factor unterscheidet "einen Streitpunkt" von "genereller Unsicherheit"
- Signal-basiertes Halting stoppt frueher bei gleicher oder besserer Qualitaet

## 6. Experiment E: Diversity-Enforcement

Kernfrage: Ohne Repulsion -- kollabieren die Worker?

```python
# Vergleich: diversity_strength = 0 vs. 0.01, 0.05, 0.1, 0.5
# Bei Fixpunkt UND Energie
# Messen:
# - Paarweise Cosine Similarity zwischen Workern ueber Iterationen
# - Konvergiert das System noch wenn Diversity erzwungen wird?
# - Sweet Spot: Genug Diversity um nicht zu kollabieren,
#   wenig genug um noch zu konvergieren

# Collapse-Metrik: Durchschnittliche paarweise Cosine Similarity
# > 0.95 = Kollaps (Worker praktisch identisch)
# < 0.3 = Zu viel Repulsion (keine Konvergenz)
```

### Erwartetes Ergebnis (aus PLR und Coda-Forno)
- Ohne Enforcement: Diversity zerfaellt exponentiell mit Iterationstiefe
- Mit Enforcement: Konvergenz ist langsamer aber Worker bleiben differenziert
- Energieminimierung handelt den Trade-off natuerlicher (Repulsion als Teil der Energie)

## 7. Experiment F: Semantische vs. zufaellige Vektoren

Kernfrage: Verhält sich realistisch strukturierter Input anders?

```python
# Vergleich: Zufaellige Gaussian-Vektoren vs. Cluster-strukturierte Vektoren
# (Fallback fuer echte Embeddings) vs. echte GloVe-Embeddings (wenn verfuegbar)
#
# Hypothese: Semantische Struktur veraendert Konvergenzverhalten,
# weil reale Daten nicht gleichverteilt im Raum liegen sondern
# auf niedrigdimensionalen Mannigfaltigkeiten.
```

## 8. Zusammenfassung & Entscheidung

Am Ende des Notebooks:

```python
# Automatisierte Zusammenfassung:
summary = {
    "convergence_works": bool,  # Konvergiert ueberhaupt?
    "best_mechanism": str,      # "fixpoint" oder "energy"
    "best_merge": str,          # Welche Strategie?
    "orthogonality_threshold": float,  # Ab wann bricht es?
    "diversity_needed": bool,   # Braucht man Enforcement?
    "signal_metrics_useful": bool,  # Tracken sie Konvergenz besser?
    "best_halting": str,        # "delta" oder "signal_based"
    "semantic_difference": bool,  # Veraendert semantische Struktur das Ergebnis?
    "recommendation": str,      # "proceed" / "revise" / "stop"
}
```

### Entscheidungsmatrix

```
Konvergenz    Signal-Metriken   Diversity    -> Naechster Schritt
--------------------------------------------------------------------
Ja            Nuetzlich         Noetig       -> Sim 2 (Probing), voller Erfolg
Ja            Nuetzlich         Nicht noetig -> Sim 2, aber Diversity-Frage offen
Ja            Nicht besser      Egal         -> Sim 2, aber unique Beitrag schwaecher
Nein (Fixp.)  -                 -            -> Nur Energie weiter, Sim 2
Nein (beides) -                 -            -> Fundamentalproblem, Ansatz ueberdenken
```

## Implementierungsreihenfolge

Nicht alles auf einmal. Reihenfolge nach absteigendem Informationsgewinn:

1. **Tag 1**: Setup + Experiment A (Konvergiert es ueberhaupt?)
   - Wenn NEIN bei exakter Orthogonalitaet: STOP, fundamentales Problem
   - Wenn JA: weiter

2. **Tag 1-2**: Experiment B (Orthogonalitaets-Sweep) + C (Merge-Vergleich)
   - Identifiziert den robustesten Merge und die Orthogonalitaets-Grenze

3. **Tag 2**: Experiment D (Signalmetriken)
   - DER zentrale Test fuer den unique Beitrag
   - Ergebnis bestimmt ob das Paper-Material ist oder nur Methodik

4. **Tag 2-3**: Experiment E (Diversity) + F (Semantische Vektoren)
   - Feintuning und Realismus-Check

## Verbindung zu den PDA-Dokumenten

- research-synthesis.md: Empfohlene Architektur (EBT, PEGO, PLR) fliesst in Experiment-Design
- pda-simulations-roadmap.md: Sim 1 Abschnitt ist die Kurzfassung dieses Plans
- parallel-deliberation-architecture-2.md: Signalmetriken-Abschnitt definiert SNR/Kohaerenz/Crest Factor
- n-pda-native-architecture.md: Orthogonale Unterraeume + DEQ -> jetzt EBT

## Notizen

- Kein GPU noetig. Alles CPU-basiert, numpy/torch ohne CUDA.
- Reproduzierbar: Fester Seed, alle Parameter dokumentiert.
- Plots: Matplotlib + Seaborn, gespeichert als PNG fuer Dokumentation.
- Laufzeit geschaetzt: <1 Stunde fuer den vollstaendigen Sweep auf einem normalen Laptop.
