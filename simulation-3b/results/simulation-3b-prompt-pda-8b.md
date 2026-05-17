# Simulation 3b: Prompt-Level PDA

In dieser Simulation testen wir die Hypothese, dass parallele Deliberation aus verschiedenen Perspektiven (Prompt-PDA) die Antwortqualitaet eines LLMs verbessert.


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import os

# Pfad zu den Modulen
sys.path.append('.')

from sim3b_perspectives import generate_perspectives, PERSPECTIVES
from sim3b_merge import merge_llm_synthesis, merge_logit_average, merge_majority_vote
from sim3b_eval import compute_diversity, compute_quality_llm, compare_with_baselines

# Modell laden
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model_id = "Qwen/Qwen3-8B"
print(f"Loading model {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print("Model loaded.")
```

    Loading model Qwen/Qwen3-8B...



    Loading checkpoint shards:   0%|          | 0/5 [00:00<?, ?it/s]


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

    /home/t0bybr/share/simulation-1/.venv/lib/python3.13/site-packages/bitsandbytes/backends/cuda/ops.py:468: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
      torch._check_is_size(blocksize)


    --- ANALYTISCH ---
    Der Himmel erscheint blau, weil das Licht der Sonne mit der Atmosphäre der Erde interagiert. Um das genauer zu verstehen, können wir das Problem in folgende Teilschritte zerlegen:
    
    ---
    
    ### **1. Licht...
    
    --- INTUITIV ---
    Okay, let's see. The user wants to explain why the sky is blue to a child. They mentioned using simple language and clear images. So first, I need to break down the science behind the blue sky in an e...
    
    --- KRITISCH ---
    Die gängige Erklärung dafür, warum der Himmel blau ist, lautet: **Die Sonnenstrahlung wird durch die Atmosphäre zerstreut, und die kurzwellige blaue Farbe wird stärker zerstreut als die langwellige ro...
    
    --- KREATIV ---
    Der Himmel ist blau, wie ein Bild, das man aus der Perspektive eines Malers sieht, der seine Farben in der falschen Reihenfolge aufträgt: die Sonne ist die Kamera, die die Welt in einen Ton schwenkt, ...
    
    --- PRAKTISCH ---
    Der Himmel erscheint blau, weil die **Lichtwellen unterschiedlicher Längen** in der Atmosphäre **unterschiedlich stark gestreut** werden – ein Effekt, der als **Rayleigh-Streuung** bekannt ist.
    
    ### W...
    
    Output-Diversitaet (BLEU-Distanz): 0.9584


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
    Der Himmel erscheint blau, und zwar aus einem wundersamen Zusammenwirken von Licht, Luft und der Art, wie wir die Welt wahrnehmen. Um es einfach und klar zu erklären, können wir uns vorstellen, dass die Sonne ein riesiges Licht-Feuer ist, das alle Farben in sich trägt – wie ein buntes Lichtbogen. Dieses Licht breitet sich durch die Atmosphäre der Erde aus, und dabei passiert etwas Besonderes:
    
    ### 🌈 **Was passiert mit dem Sonnenlicht?**
    
    1. **Licht besteht aus verschiedenen Farben**  
       - Wenn du einen Regenbogen siehst, weißt du, dass das Licht aus vielen Farben besteht – von Rot bis Violett.  
       - Jede Farbe hat eine andere **Wellenlänge** – das ist wie die Länge eines Schlauches.  
       - **Blaues Licht** hat eine **kürzere Wellenlänge** als **rotes Licht**.
    
    2. **Die Luft streut das Licht**  
       - Als das Licht durch die Luft geht, **streuen** die Moleküle in der Luft (wie Stickstoff und Sauerstoff) das Licht.  
       - Dabei **brechen** sie das Licht in alle Richtungen – wie wenn man einen Kugelspiegel benutzt.
    
    Logit-Average Merge:
    Der Himmel ist blau, weil die Sonnenstrahlen, die durch die Atmosphäre der Erde reisen, mit der Luft wechselwirken. Das ist ein Vorgang, der sich **Ray


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
      PDA Merged Score:  0.84
    
    Task: Reasoning
      Single Pass Score: 0.59
      PDA Merged Score:  0.92
    
    Task: Erklaerung
      Single Pass Score: 0.72
      PDA Merged Score:  0.76
    
    Task: Kreativ
      Single Pass Score: 0.73
      PDA Merged Score:  0.92
    


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
      Score nach Runde 1: 0.76
    Runde 2...
      Score nach Runde 2: 0.76
    
    Finaler Output:
    ## 🌟 **Synthese: Die Relativitätstheorie – Eine umfassende, präzise und verständliche Erklärung**
    
    Die **Relativitätstheorie** ist eine der grundlegenden Theorien der modernen Physik und wurde von **Albert Einstein** im frühen 20. Jahrhundert entwickelt. Sie besteht aus zwei Hauptteilen: der **Speziellen Relativitätstheorie (SRT)** aus dem Jahr **1905** und der **Allgemeinen Relativitätstheorie (ART)** aus dem Jahr **1915**. Beide Theorien haben unser Verständnis von **Raum, Zeit, Bewegung und Gravitation** radikal verändert und bis heute **wissenschaftliche Grundlagen und Technologien** beeinflusst.
    
    ---
    
    ## 📌 1. **Spezielle Relativitätstheorie (SRT)**
    
    ### **Grundprinzip:**
    Die **Prinzip der Relativität** besagt, dass **alle physikalischen Gesetze in allen gleichförmig bewegten Bezugssystemen gleich sind**. Es gibt **kein bevorzugtes Bezugssystem**, das als "ruhend" oder "absolut" gilt.
    
    ### **Zentrale Erkenntnis:**


## Exp E: Vergleich mit Baselines

Zusammenfassender Vergleich.


```python
problem = "Was sind die Vor- und Nachteile der Kernkraft?"
res = compare_with_baselines(model, tokenizer, problem, n_perspectives=5)
print(f"Baseline (Single Pass): {res['score_single']:.2f}")
print(f"PDA (n=5 Merge):       {res['score_pda']:.2f}")
```

    Baseline (Single Pass): 0.75
    PDA (n=5 Merge):       0.86



```python

```
