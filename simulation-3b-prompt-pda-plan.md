# Simulation 3b: Prompt-Level PDA

Status: Planung
Datum: 2026-04-01
Autoren: Toby, Claude

## Motivation

Sim 3 hat gezeigt: PDA als Retrofit auf bestehende Modelle funktioniert auf
Aktivierungsebene (CosSim >0.97), aber der Output degradiert trotzdem.
3% Abweichung in den Aktivierungen reicht um den Text zu zerstoeren.

Prompt-PDA testet die Kernhypothese auf einer anderen Ebene:
**Bringt parallele Deliberation aus verschiedenen Perspektiven bessere Ergebnisse
als ein einzelner Durchlauf?**

Kein Eingriff in Modell-Internals. Nur: gleiches Problem, verschiedene Perspektiven,
Ergebnisse mergen. Wenn das auf Prompt-Ebene funktioniert, ist das die staerkste
Motivation fuer ein nativ paralleles Modell (Sim 4).

## Konzept

```
Problem P
    |
    +---> Perspektive 1 (z.B. "analytisch")  --> Output O1
    +---> Perspektive 2 (z.B. "kreativ")     --> Output O2
    +---> Perspektive 3 (z.B. "kritisch")    --> Output O3
    |
    v
  Merge(O1, O2, O3) --> Finaler Output
    |
    v
  [Optional: Loop — Merged Output als neuer Input]
```

## Experimente

### Exp A: Perspektiv-Design

Welche Perspektiven bringen tatsaechlich verschiedene Outputs?

Setup:
- Festes Problem (z.B. "Erklaere warum der Himmel blau ist")
- n=3-5 verschiedene System-Prompts die verschiedene Denkstile erzwingen
- Modell: Qwen3-0.6B lokal (schnell, guenstig, reproduzierbar)

Perspektiv-Kandidaten:
1. **Analytisch**: "Zerlege das Problem in Teilschritte. Arbeite systematisch."
2. **Intuitiv**: "Erklaere es so, wie du es einem Kind erklaeren wuerdest."
3. **Kritisch**: "Hinterfrage die gaengige Erklaerung. Was koennte falsch sein?"
4. **Kreativ**: "Finde eine ungewoehnliche Analogie oder Metapher."
5. **Praktisch**: "Konzentriere dich auf das, was man beobachten und testen kann."

Metriken:
- Output-Diversitaet: Wie unterschiedlich sind die n Outputs? (BLEU zwischen Paaren)
- Semantische Abdeckung: Wie viele verschiedene Aspekte werden angesprochen?
- Qualitaet pro Perspektive: Ist eine Perspektive konsistent besser?

### Exp B: Merge-Strategien

Wie kombiniert man n Text-Outputs sinnvoll?

Strategien:
1. **LLM-Synthese**: Ein weiterer LLM-Call der alle n Outputs als Input bekommt
   und eine Synthese erstellt. ("Hier sind 3 Antworten auf die gleiche Frage.
   Erstelle eine optimale Antwort die das Beste aus allen kombiniert.")
2. **Logit-Average**: Auf Token-Ebene die Logits der n Passes mitteln
   (nur lokal moeglich, braucht Zugang zu Logits)
3. **Majority Voting**: Fuer Aufgaben mit klarer Antwort (Mathe, Logik)
4. **Iterative Refinement**: Output von Runde 1 als Kontext fuer Runde 2
   ("Hier ist ein erster Entwurf. Verbessere ihn aus Perspektive X.")

Fuer jede Strategie: Vergleich mit Single-Pass Baseline.

### Exp C: Aufgabentypen

Verschiedene Aufgaben profitieren unterschiedlich von Perspektiv-Vielfalt.

Aufgaben (aufsteigende erwartete PDA-Relevanz):
1. **Faktenabfrage**: "Was ist die Hauptstadt von Frankreich?"
   -> Erwartung: PDA hilft nicht (eine Perspektive reicht)
2. **Einfaches Reasoning**: "Wenn A>B und B>C, was folgt?"
   -> Erwartung: PDA hilft wenig
3. **Komplexes Reasoning**: GSM8K-artige Mathe-Aufgaben
   -> Erwartung: PDA hilft (verschiedene Loesungswege)
4. **Erklaerung**: "Erklaere Quantenverschraenkung"
   -> Erwartung: PDA hilft (verschiedene Analogien, Aspekte)
5. **Kreatives Schreiben**: "Schreibe den Anfang einer Geschichte ueber..."
   -> Erwartung: PDA hilft stark (verschiedene Stile, Ideen)
6. **Analyse**: "Was sind die Vor- und Nachteile von X?"
   -> Erwartung: PDA hilft stark (systematische Perspektiv-Abdeckung)

Pro Aufgabentyp: 5-10 Beispiele, n=3 Perspektiven, alle Merge-Strategien.

### Exp D: Loop-Test (iterative Deliberation)

Der eigentliche PDA-Test: nicht nur einmal parallel, sondern iterativ.

Setup:
- Runde 1: n Perspektiven generieren Outputs
- Merge: Synthese erstellen
- Runde 2: n Perspektiven bekommen die Synthese als Kontext und verbessern
- Merge: Neue Synthese
- ... bis Runde r

Fragen:
- Konvergiert die Qualitaet? (Wird jede Runde besser?)
- Ab wann kommt nichts Neues mehr? (Halting-Kriterium)
- Korreliert das mit den Signalmetriken aus Sim 1?
  (SNR/Coherence auf die Embedding-Vektoren der Outputs anwenden)

Variablen:
- n = 2, 3, 5 Perspektiven
- r = 1, 2, 3, 5 Runden
- Aufgabentyp: Reasoning, Erklaerung, Analyse

### Exp E: Vergleich mit Baselines

PDA muss sich gegen existierende Methoden behaupten:
1. **Single Pass**: Eine Perspektive, kein Merge
2. **Best-of-N**: N Outputs generieren, den besten waehlen (braucht Evaluator)
3. **Self-Consistency** (Wang et al.): Mehrere Chain-of-Thought Pfade, Majority Vote
4. **Debate** (Du et al.): Zwei Modelle argumentieren gegeneinander
5. **Reflexion**: Ein Pass + "Ueberpruefe deine Antwort"

Wenn PDA diese Baselines nicht schlaegt, ist parallele Deliberation auf Prompt-Ebene
nicht besser als existierende Techniken.

## Technisches Setup

### Modell
- Primaer: Qwen3-0.6B lokal (TransformerLens oder HuggingFace generate)
- Optional: Qwen3-8B (4-bit) fuer Vergleich ob Groesse den Effekt aendert
- Optional: API-Modell (Claude/GPT) fuer qualitative Validierung

### Evaluation
- **Automatisch**: BLEU, ROUGE, BERTScore zwischen PDA-Output und Referenz
- **LLM-as-Judge**: Groesseres Modell bewertet Qualitaet (falls API verfuegbar)
- **Aufgabenspezifisch**: Accuracy fuer Mathe/Logik, Diversity-Score fuer Kreativ

### Dateien
- `sim3b_perspectives.py` -- Perspektiv-Prompts und Generierung
- `sim3b_merge.py` -- Merge-Strategien
- `sim3b_eval.py` -- Evaluation und Vergleich
- `simulation-3b-prompt-pda.ipynb` -- Hauptnotebook

## Entscheidungskriterien

### Positiv
PDA-Merge schlaegt Single-Pass UND Best-of-N bei mindestens 2 von 4
Aufgabentypen (Reasoning, Erklaerung, Kreativ, Analyse). Iterativer Loop
zeigt messbare Verbesserung ueber Runden.
-> Starke Motivation fuer Sim 4 (natives PDA-Modell).

### Gemischt
PDA-Merge schlaegt Single-Pass, aber nicht Best-of-N oder Self-Consistency.
-> Parallele Perspektiven helfen, aber der Merge ist das Problem.
   Sim 4 koennte das loesen (interner Merge statt Text-Merge).

### Negativ
PDA-Merge schlaegt Single-Pass nicht konsistent.
-> Parallele Deliberation bringt auf dieser Ebene nichts.
   Grundannahme ueberdenken.

## Abhaengigkeiten
- Kein Code aus Sim 1-3 noetig (komplett eigenstaendig)
- Signalmetriken aus Sim 1 optional fuer Loop-Analyse (Exp D)

## Verbindung zur Roadmap

```
Sim 1:  Mathe              -> POSITIV
Sim 2:  Probing             -> NEGATIV (revidiert mit Mean-Sep)
Sim 2b: Mean-Separated      -> POSITIV
Sim 3:  PDA Retrofit        -> NEGATIV (CosSim taeuscht)
Sim 3b: Prompt-PDA          -> [diese Simulation]
Sim 4:  Toy from Scratch    -> [nach Sim 3b, falls positiv]
```
