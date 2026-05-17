# Simulation 3: PDA auf bestehendem Modell

Status: Planung
Datum: 2026-04-01
Autoren: Toby, Claude
Modell: Qwen3-0.6B (TransformerLens), optional Qwen3-8B (4-bit HF)
Hardware: RTX 3060 (12GB)

## Kontext

### Sim 1 (Mathematische Konvergenz)
- Energieminimierung funktioniert (lr=0.1, 11 Iterationen, n=3-5)
- Fixpunkt-Iteration tot
- Signalmetriken (SNR, Phase Coherence, Crest Factor) > Delta als Halting

### Sim 2 (Probing) + Sim 2b (Mean-Separated)
- Original Sim 2: NEGATIV (CosSim 0.31, Subspaces verflochten)
- Sim 2b Kernfund: **Eine dominante Richtung maskiert die gesamte Subspace-Struktur**
  - Layer 7-21: dominante Richtung = 99-100% der Varianz
  - Nach Abzug: PR springt von ~1 auf ~120
  - Parallele Verarbeitung mit Mean-Separation: CosSim 0.60 -> 0.93 (k=2)
- Sim 2 Ergebnis revidiert: **POSITIV mit Korrektur**
- Die Korrektur (DC-Offset entfernen) kommt direkt aus Tobys Audio-Engineering-Intuition

### Was sich aendert
Die urspruengliche Sim 3 war als Toy-Modell from scratch geplant, weil Sim 2
negativ war. Da Sim 2b zeigt, dass bestehende Modelle funktionieren, wird Sim 3
zu einer systematischen Exploration von PDA auf bestehenden Modellen. Das
Toy-Modell (jetzt Sim 4) wird zum Vergleichsexperiment.

## Kernfragen

1. **Multi-Layer**: Funktioniert Mean-Separated PDA ueber mehrere Layer hinweg,
   nicht nur fuer einen einzelnen Layer-Uebergang?
2. **Iterative Refinement**: Kann die Energieminimierung aus Sim 1 die
   parallelen Subspaces auf echten Aktivierungen verbessern?
3. **Output-Qualitaet**: Wie gut ist der tatsaechliche Text-Output, nicht nur
   die Aktivierungs-Aehnlichkeit?
4. **Layer-Selektion**: Welche Layer profitieren am meisten von paralleler
   Verarbeitung? (Sim 2b: Layer 1 bester PR, aber mittlere Layer reichste Struktur
   nach Mean-Removal)
5. **Skalierung**: Verhaelt sich Qwen3-8B anders als 0.6B bei Multi-Layer PDA?

## Experimente

### Exp A: Multi-Layer PDA Pipeline

Sim 2b hat gezeigt: ein Layer-Uebergang funktioniert. Aber ein einzelner
Layer-Uebergang ist keine echte parallele Deliberation. Die Frage ist ob
man mehrere Layer hintereinander parallel verarbeiten kann.

Setup:
- Mean-Separated Zerlegung an Layer L
- Jeden Subspace durch Layer L+1 schicken (wie Sim 2b)
- Am Output wieder Mean-Separated zerlegen
- Durch Layer L+2 schicken
- ... bis Layer L+n
- Am Ende mergen und mit normalem Forward Pass vergleichen

Variablen:
- Start-Layer: 1, n/4, n/2 (basierend auf Sim 2b PR-Ergebnisse)
- Anzahl Layer: 1, 2, 3, 5, 10
- k Subspaces: 2, 3, 5
- Re-Separation: Mean-Direction nach jedem Layer neu berechnen vs. einmal am Anfang

Metriken:
- CosSim an jedem Zwischen-Layer (Degradationskurve)
- KL-Divergenz der finalen Logits
- Token-Accuracy (Top-1 Match)

Erwartung: Degradation pro Layer, aber die Frage ist wie schnell. Wenn
CosSim nach 5 Layern noch >0.8 ist, funktioniert Multi-Layer PDA.

### Exp B: Energieminimierung auf echten Subspaces

Sim 1 hat gezeigt: Energieminimierung konvergiert in 11 Iterationen
(bei synthetischen Vektoren). Jetzt testen wir ob das auch auf echte
Aktivierungen uebertragbar ist.

Setup:
- Mean-Separated Subspaces aus Layer L extrahieren
- Energieminimierung (iterate_energy aus Sim 1, adaptiert) auf die
  Subspace-Outputs anwenden
- Vergleich: mit vs. ohne Energieminimierung

Adaptierungen fuer echte Daten:
- Energy-Function: paarweise Distanz der Subspace-Outputs nach Layer L+1
- Learning Rate: lr=0.1 als Start (Sim 1 Optimum), sweep 0.01-0.5
- Halting: SNR-basiert (Sim 1 Erkenntnis: Signalmetriken > Delta)

Metriken:
- CosSim vorher/nachher (Energieminimierung vs. einfacher Average-Merge)
- Anzahl Iterationen bis Halting
- SNR-Verlauf ueber Iterationen

Erwartung: Energieminimierung sollte den Merge verbessern, besonders bei
hoeherem k (wo Average-Merge schlechter wird).

### Exp C: Text-Output-Qualitaet

Die bisherigen Metriken (CosSim, MSE) messen Aktivierungs-Aehnlichkeit.
Das ist notwendig aber nicht hinreichend. Hier testen wir was wirklich zaehlt:
produziert PDA sinnvollen Text?

Setup:
- Verschiedene Prompts (Fakten, Reasoning, kreativ)
- Forward Pass mit PDA (Mean-Separated, bester k und Layer-Bereich aus Exp A)
- Generierte Tokens vergleichen: PDA vs. Normal

Metriken:
- Token-fuer-Token Match: Wie viele der Top-10 Tokens stimmen ueberein?
- BLEU/ROUGE zwischen PDA-generiertem und normal generiertem Text
- Perplexity des PDA-Outputs (gemessen am Original-Modell)
- Qualitative Beispiele: 5-10 Prompts, PDA-Output vs. Normal-Output nebeneinander

Erwartung: Bei k=2 und wenigen Layern sollte der Output fast identisch sein.
Bei hoeherem k erwarten wir Divergenz — die Frage ist ob der PDA-Output
immer noch kohaerenter Text ist oder Unsinn.

### Exp D: Layer-Profiling

Sim 2b hat nur ausgewaehlte Layer getestet. Hier ein systematisches
Profiling aller Layer:

Setup:
- Fuer jeden Layer: Mean-Separated PR, dominante Varianz, CosSim bei k=2
- Heatmap: Layer-Position vs. PDA-Qualitaet

Erwartung: Es gibt einen "Sweet Spot" von Layern (vermutlich mittlere
Layer nach Sim 2b), wo die Subspace-Struktur am reichsten ist UND
parallele Verarbeitung am besten funktioniert.

### Exp E: Signalmetriken als Qualitaetspraediktor

Verbindung zwischen Sim-1-Signalmetriken und Sim-3-Output-Qualitaet.

Setup:
- Fuer jede Exp-A/B Konfiguration: SNR, Phase Coherence, Crest Factor berechnen
- Korrelation mit Output-Metriken (KL-Div, Token-Accuracy, Perplexity)

Kernfrage: Koennen die Signalmetriken vorhersagen, welche PDA-Konfiguration
guten Output liefert, OHNE den Output zu berechnen? Das waere der Schluessel
fuer ein adaptives Halting-Kriterium.

## Technisches Setup

### Umgebung
```bash
# Bestehendes venv, transformers 4.5.7
pip install transformer-lens transformers==4.5.7 torch einops jaxtyping
```

### Dateien
- `sim3_pipeline.py` -- Multi-Layer PDA Pipeline mit Mean-Separation
- `sim3_energy.py` -- Adaptierte Energieminimierung fuer echte Aktivierungen
- `sim3_metrics.py` -- Output-Qualitaetsmetriken (BLEU, Perplexity, Token-Match)
- `simulation-3-pda-existing.ipynb` -- Hauptnotebook Exp A-E

### Abhaengigkeiten
- `sim1_metrics.py` -- SNR, Phase Coherence, Crest Factor
- `sim1_helpers.py` -- iterate_energy (Basis fuer Exp B)
- `sim2_helpers.py` -- extract_activations, parallel_forward
- `simulation-2b` -- Mean-Separation Ansatz (parallel_forward_mean_separated)

## Entscheidungskriterien

### Stark positiv
Multi-Layer PDA ueber 3+ Layer mit CosSim >0.8, generierter Text kohaerenter
und sinnvoll, Energieminimierung verbessert den Merge messbar.
-> PDA auf bestehenden Modellen ist viable. Sim 4 (Toy-Modell) als Vergleich.

### Positiv
Ein-Layer PDA funktioniert gut, Multi-Layer degradiert aber schnell.
Generierter Text ist akzeptabel bei k=2-3.
-> PDA als "shallow parallelism" viable, tiefe Deliberation braucht Training (Sim 4).

### Negativ
Auch Ein-Layer PDA produziert keinen sinnvollen Text-Output trotz hoher CosSim.
-> CosSim auf Aktivierungen tauescht. Sim 2b Ergebnis war ein Artefakt.
   Zurueck zu Sim 4 (from scratch).

## Offene Fragen

- Ist die dominante Richtung pro Layer stabil ueber verschiedene Inputs?
  Wenn ja: kann man sie einmal berechnen und cachen.
  Wenn nein: muss sie pro Forward Pass berechnet werden (teurer).
- Wie verhaelt sich die dominante Richtung zur Layer Norm?
  Moeglicherweise IST sie ein Layer-Norm-Artefakt.
- Fuer Sim 4: Trainiert ein Modell mit PDA-Loss diese Separation automatisch,
  oder braucht es explizites Enforcement?

## Verbindung zur Roadmap

```
Sim 1: Mathe            -> POSITIV (Energieminimierung funktioniert)
Sim 2: Probing           -> NEGATIV (Subspaces verflochten)
Sim 2b: Mean-Separated   -> POSITIV (DC-Offset war das Problem)
Sim 3: PDA auf Existing  -> [diese Simulation]
Sim 4: Toy from Scratch  -> [Vergleichsexperiment, nach Sim 3]
```
