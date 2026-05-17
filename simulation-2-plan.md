# Simulation 2: Probing an bestehendem Modell

Status: Planung
Datum: 2026-03-29
Autoren: Toby, Claude
Modell: Qwen3-0.6B (Qwen/Qwen3-0.6B via HuggingFace)
Hardware: RTX 3060 (12GB)
Tools: TransformerLens, PyTorch, numpy

## Kontext aus Simulation 1

- Energieminimierung funktioniert (lr=0.1, 11 Iterationen, n=3-5)
- Fixpunkt ist tot
- Signalmetriken (SNR, Phase Coherence, Crest Factor) > Delta als Halting-Kriterium
- Orthogonalitaet hilfreich aber nicht zwingend
- Diversity-Constraint greift erst bei nicht-orthogonalen Raeumen (= echte Aktivierungen)

## Kernfragen

1. **Subspace-Struktur**: Zerfallen Transformer-Aktivierungen natuerlich in trennbare
   Subspaces? Wie viele dominante Komponenten gibt es?
2. **Rekonstruktion**: Wenn man Aktivierungen in k Subspaces zerlegt und wieder
   zusammensetzt — wie viel Information geht verloren?
3. **Semantische Trennbarkeit**: Kodieren verschiedene Subspaces verschiedene
   Aspekte (Syntax vs. Semantik vs. Position)?
4. **Parallele Verarbeitung**: Kann man die zerlegten Subspaces getrennt durch
   nachfolgende Layer schicken und das Ergebnis mergen?
5. **Signalmetriken auf echten Daten**: Verhalten sich SNR/Coherence/Crest Factor
   auf echten Aktivierungen informativ?

## Experimente

### Exp A: Activation Landscape (Grundlagen)

Aktivierungen aus allen Layern extrahieren fuer diverse Prompts.
Analyse der Grundstruktur:
- Singular Value Spectrum pro Layer (wie schnell fallen Eigenwerte ab?)
- Effektive Dimensionalitaet (Anzahl Komponenten fuer 90%/95%/99% Varianz)
- Vergleich: fruehe vs. mittlere vs. spaete Layer
- Vergleich: Fakten-Prompt vs. Reasoning-Prompt vs. kreativer Prompt

**Erwartung**: Mittlere Layer haben die reichste Struktur (weder zu nah am
Input-Embedding noch zu nah am Output-Unembedding).

### Exp B: Subspace-Zerlegung und Rekonstruktion

Aktivierungen per SVD/PCA in k Komponenten zerlegen, Top-k behalten,
rekonstruieren, durch restliche Layer schicken, Output vergleichen.

- k = 2, 3, 5, 10, 50% der Dimension
- Layer: 3 ausgewaehlte (frueh, mitte, spaet) basierend auf Exp A
- Metrik: KL-Divergenz der Output-Logits (Original vs. rekonstruiert)
- Metrik: Token-Accuracy (stimmt das Top-1-Token noch ueberein?)

**Erwartung**: Bei k=5-10 sollte Rekonstruktion in mittleren Layern
akzeptabel sein (<10% KL-Divergenz). Fruehe/spaete Layer brauchen mehr k.

### Exp C: Semantische Analyse der Subspaces

Fuer die dominanten Subspaces aus Exp B:
- Welche Tokens/Positionen aktivieren welchen Subspace am staerksten?
- Korrelation zwischen Subspace-Aktivierung und linguistischen Features
  (POS-Tags, Satzposition, semantische Rolle)
- Sind die Subspaces konsistent ueber verschiedene Prompts?

**Erwartung**: Zumindest teilweise interpretierbare Subspaces. Wenn
komplett uninterpretierbar: PDA-Annahme der "Perspektiven" ist fragwuerdig.

### Exp D: Parallele Subspace-Verarbeitung (der PDA-Test)

Der eigentliche Test: Aktivierungen zerlegen, GETRENNT durch den naechsten
Layer schicken, dann mergen (mit Sim-1-Erkenntnissen).

- Zerlegung in k=3-5 Subspaces (SVD) an Layer L
- Jeden Subspace einzeln durch Layer L+1 schicken
- Ergebnisse mergen (average, da Merge-Strategie egal laut Sim 1)
- Energieminimierung anwenden (iterate_energy aus Sim 1, adaptiert)
- Vergleich mit normalem Forward Pass

**Erwartung**: Output-Qualitaet sinkt, aber die Frage ist WIE STARK.
Wenn <20% Degradation bei k=3-5: PDA-Grundannahme funktioniert auf
echten Aktivierungen. Wenn >50%: Subspaces sind zu verflochten.

### Exp E: Signalmetriken auf echten Aktivierungen

Sim-1-Metriken (SNR, Phase Coherence, Crest Factor) auf die parallelen
Verarbeitungspfade aus Exp D anwenden.

- Tracken die Metriken die Qualitaet der Rekonstruktion?
- Korrelation SNR vs. KL-Divergenz?
- Kann die Halting-Entscheidung aus Sim 1 sinnvoll auf echte Daten
  angewendet werden?

**Erwartung**: SNR und Phase Coherence sollten mit Output-Qualitaet
korrelieren. Crest Factor zeigt, wo die groessten Diskrepanzen sind.

## Technisches Setup

### Umgebung
```bash
# Im bestehenden venv
pip install transformer-lens transformers torch einops jaxtyping
```

### Modell
```python
import transformer_lens as tl
model = tl.HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B")
```

### Activation Caching
```python
# TransformerLens cached run
logits, cache = model.run_with_cache(tokens)
# cache["blocks.{layer}.hook_resid_post"] -> Aktivierungen nach Layer
```

### Dateien
- `sim2_helpers.py` — Zerlegung, Rekonstruktion, parallele Verarbeitung
- `sim2_metrics.py` — Adaptierte Signalmetriken fuer echte Aktivierungen
- `simulation-2-probing.ipynb` — Hauptnotebook mit Exp A-E

## Abhaengigkeiten von Sim 1
- `sim1_metrics.py` — compute_snr, compute_phase_coherence, compute_crest_factor
- `sim1_helpers.py` — iterate_energy (adaptiert fuer echte Aktivierungen)

## Entscheidungskriterium
- **Positiv**: Zerlegung in 3-5 Subspaces mit <10% KL-Divergenz in mittleren
  Layern. Signalmetriken korrelieren mit Output-Qualitaet.
- **Negativ**: Zerlegung zerstoert systematisch Information, kein Rekonstruktionspfad.
- **Differenziert**: Rekonstruktion ok, aber parallele Verarbeitung (Exp D) degradiert
  stark -> Subspaces sind trennbar aber nicht unabhaengig verarbeitbar. Dann braucht
  PDA Cross-Attention zwischen Workern (was in Sim 3 getestet wird).
