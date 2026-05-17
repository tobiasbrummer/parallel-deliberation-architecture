# PDA Experimente: Implementierungs-Spec

Zwei Experimente als erste empirische Tests fuer die Parallel Deliberation Architecture.
Beide nutzen TransformerLens und ein kleines Open-Weights-Modell.

Kontext: share/parallel-deliberation-architecture.md

## Setup (fuer beide Experimente)

### Modell
- Qwen 2.5 0.5B oder 1.5B (gut unterstuetzt in TransformerLens, laeuft lokal)
- Alternativ: Llama 3.2 1B
- Quantisierung optional, aber fp16/bf16 bevorzugt fuer saubere Aktivierungen

### Libraries
```
transformer_lens
torch
numpy
matplotlib (fuer Visualisierung)
```

### Layer-Einteilung
Das Modell hat N Layer. Fuer beide Experimente teilen wir sie in drei Zonen:
- **Embedding-Zone**: Layer 0 bis ~N/4 (Sprache -> interne Repraesentation)
- **Mittlere Zone**: Layer ~N/4 bis ~3N/4 (Kernverarbeitung)
- **Output-Zone**: Layer ~3N/4 bis N (interne Repraesentation -> Sprache)

Die genauen Grenzen sind ein Hyperparameter. Als Start: Drittelung.

---

## Experiment 1: Konsensus / Loesungsraum-Stabilitaet

### Fragestellung
Wie stabil ist der Loesungsraum eines Modells bei einer gegebenen Frage?
Kann die Varianz zwischen parallelen Durchlaeufen als Konfidenz-Signal dienen?

### Hypothese
- Einfache, eindeutige Fragen -> niedrige Varianz (hoher Konsensus)
- Mehrdeutige, offene Fragen -> hohe Varianz (niedriger Konsensus)
- Die Varianz korreliert mit der tatsaechlichen Schwierigkeit/Mehrdeutigkeit

### Methode

```python
import transformer_lens as tl
import torch
import numpy as np

model = tl.HookedTransformer.from_pretrained("Qwen/Qwen2.5-0.5B")

def measure_consensus(prompt: str, n_runs: int = 10, perturbation_scale: float = 0.01):
    """
    Fuehrt n Forward Passes mit leicht verschiedenen Startbedingungen durch.
    Misst die Varianz der Aktivierungen an jedem Layer.
    """
    tokens = model.to_tokens(prompt)
    all_activations = {}  # layer_name -> list of activation tensors

    for i in range(n_runs):
        # Hook: Kleine zufaellige Stoerung auf die Embeddings addieren
        def perturb_embed(value, hook):
            noise = torch.randn_like(value) * perturbation_scale
            return value + noise

        # Forward Pass mit Perturbation, alle Residual-Stream-Aktivierungen speichern
        logits, cache = model.run_with_cache(
            tokens,
            fwd_hooks=[("hook_embed", perturb_embed)]
        )

        for layer_idx in range(model.cfg.n_layers):
            key = f"blocks.{layer_idx}.hook_resid_post"
            if key not in all_activations:
                all_activations[key] = []
            all_activations[key].append(cache[key].detach().cpu())

    # Varianz pro Layer berechnen
    results = {}
    for key, acts in all_activations.items():
        stacked = torch.stack(acts)  # (n_runs, batch, seq, d_model)
        # Cosine Similarity zwischen allen Paaren (am letzten Token)
        last_token_acts = stacked[:, 0, -1, :]  # (n_runs, d_model)
        cos_sims = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                cos_sim = torch.nn.functional.cosine_similarity(
                    last_token_acts[i], last_token_acts[j], dim=0
                )
                cos_sims.append(cos_sim.item())
        results[key] = {
            "mean_cosine_similarity": np.mean(cos_sims),
            "std_cosine_similarity": np.std(cos_sims),
        }

    # Auch: Top-k Token-Vorhersagen vergleichen
    # (wie oft stimmen die Top-1 Predictions ueberein?)

    return results
```

### Test-Prompts

Drei Kategorien, je 3-5 Prompts:

**Eindeutig (erwarteter hoher Konsensus):**
- "The capital of France is"
- "2 + 2 ="
- "Water freezes at"

**Mehrdeutig (erwarteter niedriger Konsensus):**
- "The meaning of life is"
- "The best programming language is"
- "Love feels like"

**Reasoning (unbekannt, interessant):**
- "If all roses are flowers and some flowers fade quickly, then"
- "The opposite of the opposite of happy is"
- "A is bigger than B. B is bigger than C. Therefore"

### Metriken
1. **Cosine Similarity pro Layer** (Durchschnitt ueber alle Paare von Runs)
2. **Top-1 Agreement Rate**: Wie oft stimmt die wahrscheinlichste Token-Vorhersage ueberein?
3. **Top-5 Overlap**: Wie viel Ueberlappung in den Top-5 Vorhersagen?
4. **Layer-Profil**: Wo divergieren die Runs? Fruehe Layer? Mittlere? Spaete?

### Erwartetes Ergebnis
Ein Plot pro Prompt-Kategorie: X-Achse = Layer, Y-Achse = Cosine Similarity.
Erwartung: Eindeutige Prompts bleiben ueberall hoch. Mehrdeutige divergieren
in den mittleren Layern. Falls das so ist: Die mittleren Layer sind tatsaechlich
der Ort wo "Entscheidungen" getroffen werden, und Varianz dort ist ein
brauchbares Konfidenz-Signal.

### Hyperparameter zum Experimentieren
- `perturbation_scale`: 0.001, 0.01, 0.1 (wie stark die Stoerung?)
- `n_runs`: 10, 20, 50 (wie viele Durchlaeufe?)
- Layer-Einteilung: Drittelung vs. andere Splits

---

## Experiment 2: Parallele Verarbeitung in mittleren Layern

### Fragestellung
Bringt es etwas, zwei parallele Verarbeitungspfade durch die mittleren Layer
zu schicken und die Ergebnisse zu kombinieren -- verglichen mit einem einzelnen Pass?

### Hypothese
Kombinierte Aktivierungen aus zwei Passes mit verschiedenen "Perspektiven"
(Steering Vectors) produzieren einen Output, der sich qualitativ vom
Single-Pass unterscheidet. Ob "besser" ist offen -- "anders und koharent"
waere schon ein positives Ergebnis.

### Methode

```python
def parallel_middle_layers(
    prompt: str,
    steering_vector_a: torch.Tensor,  # (d_model,)
    steering_vector_b: torch.Tensor,  # (d_model,)
    steering_layer: int,              # Ab welchem Layer steuern
    merge_layer: int,                 # An welchem Layer zusammenfuehren
    merge_method: str = "mean",       # "mean", "weighted", "max"
    steering_scale: float = 1.0,
):
    """
    1. Forward Pass durch Embedding-Zone (gemeinsam)
    2. Ab steering_layer: Zwei parallele Passes mit verschiedenen Steering Vectors
    3. An merge_layer: Aktivierungen zusammenfuehren
    4. Ab merge_layer: Gemeinsamer Forward Pass durch Output-Zone
    """
    tokens = model.to_tokens(prompt)

    # Schritt 1: Gemeinsamer Pass bis steering_layer, Aktivierungen speichern
    _, cache_shared = model.run_with_cache(tokens)
    shared_resid = cache_shared[f"blocks.{steering_layer}.hook_resid_pre"].clone()

    # Schritt 2a: Pass A (mit Steering Vector A)
    def steer_a(value, hook):
        return value + steering_vector_a * steering_scale

    _, cache_a = model.run_with_cache(
        tokens,
        fwd_hooks=[(f"blocks.{steering_layer}.hook_resid_pre", steer_a)]
    )

    # Schritt 2b: Pass B (mit Steering Vector B)
    def steer_b(value, hook):
        return value + steering_vector_b * steering_scale

    _, cache_b = model.run_with_cache(
        tokens,
        fwd_hooks=[(f"blocks.{steering_layer}.hook_resid_pre", steer_b)]
    )

    # Schritt 3: Aktivierungen an merge_layer kombinieren
    resid_a = cache_a[f"blocks.{merge_layer}.hook_resid_post"]
    resid_b = cache_b[f"blocks.{merge_layer}.hook_resid_post"]

    if merge_method == "mean":
        merged = (resid_a + resid_b) / 2
    elif merge_method == "max":
        merged = torch.max(resid_a, resid_b)
    # Weitere Methoden: gewichteter Durchschnitt, gelernte Kombination

    # Schritt 4: Merged Aktivierungen durch restliche Layer schicken
    def inject_merged(value, hook):
        return merged

    logits_merged = model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{merge_layer}.hook_resid_post", inject_merged)]
    )

    # Vergleich: Single Pass (Baseline)
    logits_baseline = model(tokens)

    # Auch: Einzelne Passes A und B fuer Vergleich
    logits_a = model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{steering_layer}.hook_resid_pre", steer_a)]
    )
    logits_b = model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{steering_layer}.hook_resid_pre", steer_b)]
    )

    return {
        "baseline": logits_baseline,
        "pass_a": logits_a,
        "pass_b": logits_b,
        "merged": logits_merged,
    }
```

### Steering Vectors erzeugen

Zwei Ansaetze:

**Ansatz 1: Kontrastive Paare**
Steering Vectors aus kontrastiven Prompt-Paaren extrahieren (Standardmethode):
```python
# Beispiel: "analytisch" vs. "kreativ"
prompts_analytical = ["Think step by step about", "Analyze logically", ...]
prompts_creative = ["Imagine freely", "What if we consider", ...]

# Durchschnittliche Aktivierung pro Gruppe, Differenz = Steering Vector
sv_analytical = mean_activations(prompts_analytical) - mean_activations(prompts_creative)
sv_creative = -sv_analytical
```

**Ansatz 2: Zufaellige orthogonale Richtungen**
Fuer den reinen Diversitaets-Test: Zwei zufaellige, orthogonale Vektoren im
Aktivierungsraum. Testet ob JEDE Diversitaet hilft, nicht nur semantisch sinnvolle.

### Test-Prompts

Prompts wo Perspektivenvielfalt helfen koennte:
- "What are the implications of artificial general intelligence?"
- "Should we colonize Mars?"
- "Explain quantum entanglement."
- Auch: Die mehrdeutigen Prompts aus Experiment 1 (Vergleichbarkeit)

### Metriken
1. **Token-Divergenz**: Wie unterscheidet sich merged Output von Baseline?
   (Token-Level-Vergleich, Cosine Similarity der Logits)
2. **Kohaerenz**: Ist der merged Output grammatikalisch/semantisch koharent?
   (Perplexity des merged Outputs, gemessen mit dem gleichen Modell)
3. **Diversitaet**: Deckt der merged Output mehr Aspekte ab als Baseline?
   (Manuell bewerten fuer eine kleine Stichprobe)
4. **Vergleich mit Einzelpasses**: Ist merged besser als A allein oder B allein?

### Hyperparameter zum Experimentieren
- `steering_layer`: N/4, N/3, N/2 (wo anfangen zu divergieren?)
- `merge_layer`: N/2, 2N/3, 3N/4 (wo zusammenfuehren?)
- `merge_method`: mean, max, gewichtet
- `steering_scale`: 0.5, 1.0, 2.0 (wie stark die Perspektiv-Divergenz?)
- Anzahl paralleler Passes: 2, 3, 5

### Moegliche Ergebnisse und Interpretation

| Ergebnis | Interpretation |
|---|---|
| Merged Output koharent und anders als Baseline | Positiv: Parallele Verarbeitung erzeugt neue Information |
| Merged Output inkoharent | Erwartet bei naivem Merging. Zeigt dass Repraesentationen nicht einfach gemittelt werden koennen. Nichtlineares Merging noetig. |
| Merged Output = Durchschnitt von A und B auf Token-Ebene | Merging passiert zu spaet oder zu einfach. Layer-Grenzen anpassen. |
| Merged Output identisch mit Baseline | Steering Vectors zu schwach oder mittlere Layer zu robust gegen Perturbation. Scale erhoehen. |

---

## Reihenfolge

1. **Experiment 1 zuerst.** Es ist einfacher, braucht keine Steering Vectors,
   und die Ergebnisse informieren Experiment 2 (welche Layer sind interessant,
   wie robust sind die Repraesentationen).

2. **Experiment 2 mit Erkenntnissen aus 1.** Insbesondere: Die Layer, wo
   Experiment 1 die groesste Divergenz zeigt, sind gute Kandidaten fuer
   `steering_layer` und `merge_layer`.

## Hinweise fuer die Implementierung

- TransformerLens Doku: https://transformerlensorg.github.io/TransformerLens/
- `run_with_cache` gibt alle Zwischen-Aktivierungen zurueck
- `run_with_hooks` erlaubt Manipulation einzelner Layer
- Hook-Namen: `blocks.{i}.hook_resid_pre`, `blocks.{i}.hook_resid_post`,
  `blocks.{i}.attn.hook_result`, etc.
- Fuer Steering Vectors: `model.run_with_cache` auf kontrastive Prompts,
  dann Differenz der Residual-Stream-Aktivierungen am gewuenschten Layer
- GPU-Memory beachten: Bei mehreren Caches gleichzeitig kann es eng werden.
  `.detach().cpu()` fuer nicht aktiv benoetigte Tensoren.
