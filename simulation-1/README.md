# Simulation 1: Mathematische Konvergenz

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ausfuehren

```bash
jupyter notebook simulation-1-convergence.ipynb
```

Oder headless:

```bash
jupyter nbconvert --to notebook --execute simulation-1-convergence.ipynb --output executed.ipynb
```

## Dateien

- `sim1_helpers.py` -- Unterraum-Generierung, Merge-Strategien, Iterationsmechanismen
- `sim1_metrics.py` -- Signalverarbeitungs-Metriken (SNR, Phasenkohaerenz, Crest Factor)
- `simulation-1-convergence.ipynb` -- Notebook mit 6 Experimenten (A-F)
- `requirements.txt` -- Dependencies

## Experimente

- A: Basis-Konvergenz (konvergiert es?)
- B: Orthogonalitaets-Sweep (ab welchem Winkel bricht es?)
- C: Merge-Strategien (welcher Merge ist am robustesten?)
- D: Signalmetriken als Diagnostik (unique Beitrag)
- E: Diversity-Enforcement (kollabieren Worker?)
- F: Semantische vs. zufaellige Vektoren
