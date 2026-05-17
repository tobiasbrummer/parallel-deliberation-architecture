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
    <think>
    Okay, the user is asking why the sky is blue. Let me start by recalling what I know. I remember that the color of the sky is due to the way light interacts with the atmosphere. But I need to b...
    
    --- INTUITIV ---
    <think>
    Okay, the user wants to know why the sky is blue. Let me start by breaking it down. I know that the sky is blue because of the way light interacts with the atmosphere. I should explain that th...
    
    --- KRITISCH ---
    <think>
    Okay, the user is asking why the sky is blue. Let me start by recalling the basic science behind it. The atmosphere plays a role here. I know that when sunlight hits the Earth's atmosphere, it...
    
    --- KREATIV ---
    <think>
    Okay, the user is asking why the sky is blue. Let me start by recalling the basic reason. I know it's because of the way sunlight interacts with the atmosphere. The Earth's atmosphere scatters...
    
    --- PRAKTISCH ---
    <think>
    Okay, the user is asking why the sky is blue. I need to explain this in a clear and simple way. Let me start by recalling the basic reason. I remember that the sky appears blue because of the ...
    
    Output-Diversitaet (BLEU-Distanz): 0.7551


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
    Okay, the user is asking why the sky is blue. Let me start by explaining the basic reason. I know that the atmosphere scatters sunlight, and blue light is scattered more. But I need to make it simple. Maybe compare it to a filter. The sun's light passes through the atmosphere, and the blue light is scattered more, making the sky appear blue. Also, mention that the color is not just about the blue light but the fact that blue light has a longer wavelength, which is scattered more. I should keep sentences short and use examples like a prism to illustrate the scattering. Make sure to highlight that the blue light becomes dominant in the sky. That should cover it. Let me put it all together in a clear, concise way.
    </think>
    
    Der Himmel blau ist aufgrund der Interaktion zwischen Sonnenlicht und der Atmosphäre. Es gibt zwei Hauptmechanismen, die den Blau im Himmel erklären:  
    
    1. **Rayleigh-Scattering**: Licht mit kurzer Frequenz (Blau) wird durch Atmungsmoleküle und Partikel des atmosphäischen Gases auftränkt. Dieser Effekt macht den Blau den dominanten Schimmer im Himmel.  
    
    2. **Spectrum-Interaktion**: Die Sonnenlicht ist eine Vielfalt von Farben, und das Blau ist die einzige Farbe, die in der Atmosphäre besonders auftr
    
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


    The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.


      Single Pass Score: 0.50
      PDA Merged Score:  0.50
    
    Task: Reasoning
      Single Pass Score: 0.50
      PDA Merged Score:  0.50
    
    Task: Erklaerung
      Single Pass Score: 0.50
      PDA Merged Score:  0.50
    
    Task: Kreativ
      Single Pass Score: 0.50
      PDA Merged Score:  0.50
    


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
      Score nach Runde 1: 0.50
    Runde 2...
      Score nach Runde 2: 0.50
    
    Finaler Output:
    <think>
    Okay, the user wants me to improve the explanation of special relativity. Let me start by reviewing the original answer. The key points are the postulates, time dilation, length contraction, Lorentz transformations, and relativity of simultaneity. The user probably wants a more detailed, structured, and comprehensive explanation.
    
    First, I should break down the theory into clear sections. Start with the foundational postulates: the constancy of the speed of light in all inertial frames, the relativity of simultaneity, and the invariance of spacetime. Then, explain the effects: time dilation (proper time vs. moving time), length contraction (length in motion vs. rest), and the Lorentz transformations. Including examples like the moving train and the person on it would help illustrate these concepts. 
    
    I need to ensure that each point is explained clearly, avoiding jargon. Also, emphasize the theory's validity through experiments and its role in modern physics. Finally, check for any improvements in flow, clarity, and completeness. Making sure the answer is both accurate and easy to understand.


## Exp E: Vergleich mit Baselines

Zusammenfassender Vergleich.


```python
problem = "Was sind die Vor- und Nachteile der Kernkraft?"
res = compare_with_baselines(model, tokenizer, problem, n_perspectives=5)
print(f"Baseline (Single Pass): {res['score_single']:.2f}")
print(f"PDA (n=5 Merge):       {res['score_pda']:.2f}")
```

    Baseline (Single Pass): 0.50
    PDA (n=5 Merge):       0.50



```python

```
