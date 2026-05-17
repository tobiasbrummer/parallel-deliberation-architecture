# Simulation 3b: Prompt-Level PDA

In dieser Simulation testen wir die Hypothese, dass parallele Deliberation aus verschiedenen Perspektiven (Prompt-PDA) die Antwortqualitaet eines LLMs verbessert.


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Pfad zu den Modulen
sys.path.append('.')

from sim3b_perspectives import generate_perspectives, PERSPECTIVES
from sim3b_merge import merge_llm_synthesis, merge_logit_average, merge_majority_vote
from sim3b_eval import compute_diversity, compute_quality_llm, compare_with_baselines

# Modell laden
model_id = "Qwen/Qwen3-0.6B"
print(f"Loading model {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
print("Model loaded.")
```

    Loading model Qwen/Qwen3-0.6B...


    `torch_dtype` is deprecated! Use `dtype` instead!


    Model loaded.


## Exp A: Perspektiv-Design

Welche Perspektiven bringen tatsaechlich verschiedene Outputs?


```python
problem = "Erklaere warum der Himmel blau ist."
outputs = generate_perspectives(model, tokenizer, problem)

for name, text in outputs.items():
    print(f"--- {name.upper()} ---")
    print(text[:200] + "...")
    print()

div = compute_diversity(outputs)
print(f"Output-Diversitaet (BLEU-Distanz): {div:.4f}")
```

    --- ANALYTISCH ---
    Der Himmel blau ist, weil der Lichtstrahl von Sonnen und Mond auf die Erde reflektiert und diese Lichtstrahlen **einbläulenden** (blauem) Lichtabbau auslösen. Diese reflektierte Lichtstrahlen, die den...
    
    --- INTUITIV ---
    Okay, let me explain why the sky is blue to a kid. So, the sky is blue because the sun is shining on the Earth. Imagine a big, colorful picture on the ground. The sun is like a big, warm light that ma...
    
    --- KRITISCH ---
    Der Himmel ist blau, weil die Sonne die Sonnenflachheit durch das Material des Himmels schafft. Dieser Effekt wird durch die spezifischen physikalischen Eigenschaften des Himmels, wie die Tiefen und d...
    
    --- KREATIV ---
    Eine ungewöhnliche Analogie oder Metapher könnte sein:
    
    **"Der Himmel ist blau, weil es der Geist des Gottes ist, der uns den Raum im Leben bietet, in dem wir uns und die Welt um uns herum lebendig un...
    
    --- PRAKTISCH ---
    Der Himmel ist blau, weil das Licht des Sonnen und der Erde in der Atmosphäre des Himmels nicht vollständig reflektiert wird. Die spezifischen Optik des Himmels und die spezifischen Eigenschaften der ...
    
    Output-Diversitaet (BLEU-Distanz): 0.9722


## Exp B: Merge-Strategien

Wie kombiniert man n Text-Outputs sinnvoll?


```python
print("LLM-Synthese Merge:")
synth = merge_llm_synthesis(model, tokenizer, outputs, problem)
print(synth)

print("\nLogit-Average Merge:")
perspective_prompts = list(PERSPECTIVES.values())[:3]
logit_avg = merge_logit_average(model, tokenizer, problem, perspective_prompts, max_tokens=50)
print(logit_avg)
```

    LLM-Synthese Merge:
    <think>
    
    **Optimale Antwort:**
    
    Der Himmel ist blau, weil die Sonne und das Licht der Erde in der Atmosphäre des Himmels nicht vollständig reflektiert und absorbiert wird. Die spezifischen physikalischen Eigenschaften des Himmels, wie die Tiefen und die Schichtdicke, ermöglichen das Licht, das Sonnen und die Erde reflektiert und absorbiert, so, dass es den menschlichen Beobachter ansehen kann. Dieses reflektierte und absorbierte Licht wirkt so, als würde die Sonne und die Erde in den Himmel reflektiert und absorbiert. Der „Blau“ ist daher ein **Faktor der Erde**, der die Wirkung der Sonne und Mond auf den menschlichen Beobachter anzeigt.
    
    **Einschraenkungen oder Falschheiten:**
    
    1. **Die Lichtgeschwindigkeit des Lichtes:** Die Lichtgeschwindigkeit ist eine konstante, und die Erklärung der Blau ist auf die Lichtgeschwindigkeit und nicht auf die Lichtgeschwindigkeit. Dies ist falsch.
    
    2. **Die Lichtgeschwindigkeit des Lichtes:** Die Lichtgeschwindigkeit ist eine konstante, und
    
    **Perspektive kreativ:**  
    Eine ungewöhnliche Analogie oder Metaph
    
    Logit-Average Merge:
    <think>
    Okay, the user is asking why the sky is blue. Let me start by recalling what I know. I remember that the sky appears blue because of the way light interacts with the atmosphere. But I need to make sure I explain it clearly


## Exp C: Aufgabentypen

PDA-Performance auf verschiedenen Aufgaben.


```python
tasks = [
    ("Fakt", "Was ist die Hauptstadt von Frankreich?"),
    ("Reasoning", "Wenn A > B und B > C, was folgt fuer A und C?"),
    ("Erklaerung", "Erklaere Quantenverschraenkung."),
    ("Kreativ", "Schreibe den Anfang einer Geschichte ueber einen zeitreisenden Toaster.")
]

for t_type, p in tasks:
    print(f"Task: {t_type}")
    res = compare_with_baselines(model, tokenizer, p, n_perspectives=3)
    print(f"  Single Pass Score: {res['score_single']:.2f}")
    print(f"  PDA Merged Score:  {res['score_pda']:.2f}")
    print()
```

    Task: Fakt
      Single Pass Score: 0.52
      PDA Merged Score:  0.74
    
    Task: Reasoning
      Single Pass Score: 0.68
      PDA Merged Score:  0.79
    
    Task: Erklaerung
      Single Pass Score: 0.68
      PDA Merged Score:  0.76
    
    Task: Kreativ
      Single Pass Score: 0.58
      PDA Merged Score:  0.85
    


## Exp D: Iterative Deliberation (Loop-Test)

Verbessert sich die Antwort ueber mehrere Runden?


```python
def iterative_pda(model, tokenizer, problem, rounds=2):
    current_best = ""
    for r in range(rounds):
        print(f"Runde {r+1}...")
        # In Runde > 0 nutzen wir den vorherigen Output als Kontext
        contextual_problem = problem
        if current_best:
            contextual_problem += f"\n\nHier ist ein erster Entwurf: {current_best}\nVerbessere diesen Entwurf."
            
        outputs = generate_perspectives(model, tokenizer, contextual_problem, 
                                       {k: PERSPECTIVES[k] for k in ["analytisch", "kritisch", "praktisch"]})
        current_best = merge_llm_synthesis(model, tokenizer, outputs, problem)
        score = compute_quality_llm(model, tokenizer, problem, current_best)
        print(f"  Score nach Runde {r+1}: {score:.2f}")
    return current_best

final_res = iterative_pda(model, tokenizer, "Erklaere die Relativitaetstheorie.", rounds=2)
print("\nFinaler Output:")
print(final_res)
```

    Runde 1...
      Score nach Runde 1: 0.75
    Runde 2...
      Score nach Runde 2: 0.76
    
    Finaler Output:
    Die Relativitaetstheorie, entwickelt von Albert Einstein und Hermann Minkowski, ist eine grundlegende Theorie der Physik, die die Geschwindigkeitsrelationen zwischen bewegten Beobachtern und Bewegten Objekten in der Zeit und der Raum beschreibt. Sie ist zentral für die Beschreibung von Quantenmechanik und Gravitation, und sie ermöglicht die Erwiderung der Einstein-Kraft (Newtons Kraft) und das Verständnis von Quantenmechanik. Die Theorie ist nicht nur eine mathematische Verallgemeinung der Geschwindigkeitsrelationen, sondern auch eine revolutionäre Erfindung, die die Einheit der Raum-Zeit und die Raum-Kurven in einem nicht-universellen Kontext zu erklären. Sie hat die Grundlagen für moderne Physik und ermöglicht die Beschreibung von kosmischen Phänomenen und der Raum-Zeit-Struktur.


## Exp E: Vergleich mit Baselines

Zusammenfassender Vergleich.


```python
problem = "Was sind die Vor- und Nachteile der Kernkraft?"
res = compare_with_baselines(model, tokenizer, problem, n_perspectives=5)
print(f"Baseline (Single Pass): {res['score_single']:.2f}")
print(f"PDA (n=5 Merge):       {res['score_pda']:.2f}")
```

    Baseline (Single Pass): 0.92
    PDA (n=5 Merge):       0.58



```python

```
