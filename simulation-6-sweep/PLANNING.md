# Sim 6 — PDA Distillation Perspective Sweep (research-beständig)

## Kontext

- Sim 5 (Qwen3-1.7B Student, Qwen3-8B Teacher, GSM8K): +25pp, PDA-Distillation validiert
- Sim 5b (erweitert auf MATH + ARC): Generalisation bestätigt
- Sim 6 v1 (Gemma-3, multi-platform): an Infrastruktur gescheitert, nicht an Idee

Sim 6 v2 soll eine **eine** Frage sauber beantworten und publikationsfähig sein.

## Forschungsfrage

Wie verändert sich die Qualität eines PDA-distilled Students mit der Anzahl Worker-Perspektiven n? Monoton? Sweet Spot? Diminishing Returns?

## Hypothesen (falsifizierbar, vorab fixiert)

- **H1**: Für mindestens ein n ∈ {2,3,4,5} erreicht PDA-n-Distillation signifikant höhere Accuracy als CoT-Distillation auf GSM8K (p < 0.05, Bonferroni-korrigiert).
- **H2**: Die Accuracy-Kurve über n ist nicht monoton steigend — Sweet Spot liegt bei n ≤ 3.

Null-Ergebnisse (kein Unterschied, oder monoton steigend) sind valide und werden gleichrangig berichtet.

## Design

### Variablen

| Typ | Variable | Werte |
|---|---|---|
| Unabhängig | Perspektiven-Count n | 1 (CoT), 2, 3, 4, 5 |
| Kontrolliert | Teacher | Claude Opus 4.6 |
| Kontrolliert | Student | unsloth/mistral-7b-v0.3 (dense, kein sliding window, Unsloth-blessed) |
| Kontrolliert | Training-Samples | 150 korrekte Examples pro Variant (gleich viele, nicht mehr für PDA) |
| Kontrolliert | LoRA-Hyperparams | r=16, lora_alpha=16, lr=2e-4, 3 epochs, batch 2, grad_accum 4 |
| Kontrolliert | Seeds | [42, 1337, 2718] |
| Abhängig | Accuracy | GSM8K (200 Q), MATH (100 Q) |

### Compute-Matching (wichtig)

Alle 5 Varianten: **150 Training-Examples**. PDA-5 verbraucht beim *Teacher* ~5× Compute, aber der Student sieht gleich viele Trainingsdaten. Die Frage ist: Lohnt sich das zusätzliche Teacher-Compute für bessere Distillation?

### Runs

5 Varianten × 3 Seeds × 2 Benchmarks = **30 Eval-Runs**
5 Varianten × 3 Seeds = **15 Training-Runs** (Eval reused base model)

## Infrastruktur

### Platform
- RunPod, RTX 3090 / 4090 / A10 (24GB+, Ampere oder Ada)
- Pod-Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1` (Torch 2.4, vor Torch-5.0-Regression)

### Dependencies (pinned in `requirements.txt`)
```
torch==2.4.0
transformers>=4.45,<5.0
peft==0.13.2
accelerate==1.1.1
bitsandbytes==0.44.1
trl<0.9.0
unsloth (latest kompatibel mit torch 2.4)
datasets
```

### Reproducibility
- Git repo mit notebook + jsonl + requirements.txt
- Git commit SHA in results-JSON schreiben
- Seed alles: torch, numpy, random, Python-Hash
- Docker-Image-Tag dokumentieren

## Analyse-Plan (vorab festgelegt)

1. Base-Model-Accuracy pro Benchmark — Sanity-Check
2. Pro (variant, benchmark): Mean ± SE über 3 Seeds
3. Paired t-test PDA-n vs CoT, pro Benchmark
4. Bonferroni: α = 0.05/4 = 0.0125 pro Vergleich
5. Cohen's d als Effect Size
6. Plot: Accuracy vs n, Error Bars, zwei Panels (GSM8K, MATH)

## Deliverables

- `sim6_results.json` — alle 30 Eval-Runs, pro-Frage, pro-Seed
- `sim6_analysis.ipynb` — Statistics + Plot
- `RESULTS.md` — Conclusion, Effect Sizes, Limitations
- Adapter-Checkpoints unter `/workspace` (optional, groß)

## Stop-Kriterien

- Setup > 2h ohne laufendes Training → Infrastruktur neu denken, nicht weiter debuggen
- Base-Accuracy auf GSM8K < 10% → Prompt-Format oder Modell prüfen (sollte bei ~30% liegen)
- Training-Loss divergiert (steigt nach Epoche 1) → Hyperparams runter, nicht blind weiter
- Eval < 10 tok/sec auf passender GPU → Fast-Path greift nicht, Setup fixen bevor Run

## Budget

- Setup + 1 Dry-Run (1 variant, 1 seed): ~2h, $2
- Vollständiger Sweep (15 trainings + 30 evals): ~15h, $10-15
- Pessimistisch mit Debug: 20h, $20

## Open Questions (später, nicht in Sim 6)

- Token-matched Baseline: "CoT × n + Best-of-n Voting" — gleicher Teacher-Compute wie PDA-n
- Cross-Domain: Training auf Math, Eval auf Code
- Teacher-Scaling: Haiku vs Sonnet vs Opus Teacher
- Student-Scaling: 1B vs 7B vs 70B

## Pre-Registration

Dieses Dokument fixiert Hypothesen und Analyseplan **bevor** Daten gesammelt werden. Ergebnisse werden unabhängig von Outcome (auch Null-Ergebnisse) berichtet.

Commit dieses PLANNING.md vor Ausführung von Training.
