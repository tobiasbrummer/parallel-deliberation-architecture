# Parallel Deliberation Architecture

Status: Fruehe Ideenphase, noch nicht getestet.
Datum: 2026-03-26
Autoren: Toby, Claude

## Kernidee

Eine Modell-Architektur, in der Reasoning nicht sequentiell Token-fuer-Token passiert,
sondern als parallele Deliberation mehrerer Worker im Vektorraum, mit dynamischer
Parameter-Allokation aus einem geteilten Pool.

## Bibliotheksmetapher

Eine Bibliothek (der Parameterraum). Darin:
- 3 Eingangsdudes: Text -> Wort-Uebersetzung -> interne Repraesentation (Embedding-Layer)
- n Runner (Worker): Werden mit Teilfragen losgeschickt, greifen auf beliebige Regale zu
- 1 Nebentisch (Deliberationsraum): Wo Runner Ergebnisse zusammenfuehren, diskutieren, iterieren
- 1 Kontrollmechanismus: Entsteht aus der Interaktion (Konvergenz), kein separates Modul
- 3 Ausgangsdudes: Interne Repraesentation -> Wort-Uebersetzung -> Text (Unembedding-Layer)
- Spezial-Runner: Koennen "rausrennen" (Web-Search, Tool-Calls im Vektorraum)

## Unterschiede zu bestehenden Architekturen

### vs. Dense Transformer
- Dense: 1 sequentieller Forward Pass durch alle Layer, Token fuer Token
- PDA: Parallele Verarbeitung, dynamisch, iterativ, Ergebnis statt Token

### vs. MoE (Mixture of Experts)
- MoE: Router waehlt feste Experten (feste Weight-Subsets), 1 pro Token, keine Kommunikation
- PDA: Worker bekommen dynamisch die Parameter die sie brauchen. Nicht "du BIST der Mathe-Experte"
  sondern "hier ist eine Mathe-Frage, nimm dir die passenden Weights". Worker sind generisch.
  Worker kommunizieren miteinander (Deliberation).

### vs. Chain of Thought / Chain of Continuous Thought
- CoT: Sequentielles Reasoning in Sprache (Token-Output als Zwischenschritte)
- CCoT: Reasoning in kontinuierlichen Vektoren, aber sequentiell
- PDA: Reasoning in Vektoren, PARALLEL, mit Iteration

### vs. Diffusion Models
- Diffusion: Iterative Verfeinerung eines Gesamtergebnisses
- PDA: Aehnlich beim Output (Ergebnis statt Token-Sequenz), aber mit strukturierter
  Deliberation statt ungerichteter Denoising-Schritte

## Architektur-Skizze

```
Input (Text)
    |
[Embedding-Layer: 3 Schichten, Text -> interne Repraesentation]
    |
[Koordinator: Verteilt Teilfragen + Parameter an Worker]
    |
[Worker 1] [Worker 2] ... [Worker n]    <-- parallel, jeder mit dynamischem Parameter-Subset
    |          |              |
    +----------+--------------+
               |
        [Deliberationsraum]              <-- Cross-Attention zwischen Worker-Outputs
               |                             Iterativ: Ergebnisse mergen, neu verteilen
        [Halting-Mechanismus]            <-- Konvergenz messen (Veraenderungsrate)
               |                             Timeout als Fallback
        [Konsolidiertes Ergebnis]        <-- Vektor-Repraesentation, nicht Tokens
               |
[Unembedding-Layer: 3 Schichten, interne Repraesentation -> Text]
    |
Output (Ganzes Ergebnis, diffusionsartig, nicht Token-fuer-Token)
```

## Hardware-Vision

- 1 Parameter-Server (GPU mit viel VRAM): Haelt alle Weights (z.B. 100B)
- n Worker-GPUs (klein, z.B. 8GB): Generische Recheneinheiten
- Koordinator (CPU oder leichte GPU): Orchestriert Parameterverteilung

Pro Iteration bekommt jeder Worker nur die Parameter die er braucht (~8B Subset).
Skalierung ueber Breite (mehr kleine GPUs) statt Hoehe (groessere GPUs).

Flaschenhals: Bandbreite Parameter-Server -> Worker.
8B fp16 = ~16GB. NVLink: ~18ms, PCIe 5.0: ~250ms.
Mitigation: Streaming (rechnen waehrend Transfer), Lokalitaet (Parameter wiederverwenden).

## Offene Fragen

### Kritisch
1. **Kompatibilitaet**: Mittlere Layer bestehender Modelle erwarten spezifischen Input.
   Parallele Verarbeitung + Merging wird die Repraesentationen wahrscheinlich brechen.
   -> Retraining noetig, mindestens fuer den Deliberationsmechanismus.

2. **Trainingssignal**: Worauf optimiert man die Deliberation?
   Next-Token-Prediction passt nicht. Outcome-basiertes RL?
   "War dieses Deliberationsergebnis besser als ohne Deliberation?"

3. **Dynamische Parameter-Allokation**: Wie entscheidet der Koordinator, welche
   Parameter ein Worker braucht? Gelerntes Routing? Aufgabenbasierte Heuristik?

### Wichtig
4. **Konvergenz**: Wie sicherstellen, dass Deliberation terminiert?
   Halting basierend auf Veraenderungsrate + Timeout.

5. **Tool-Integration**: Tools liefern aktuell Text. Fuer Vektorraum-Integration
   braucht jedes Tool einen Encoder in den internen Repraesentationsraum.

6. **Evaluation**: Auf welchen Aufgaben misst man ob es besser ist?
   Reasoning-Benchmarks? Kreative Aufgaben? Offene Probleme?

### Erkundungswert
7. **Skalierungsverhalten**: Wird es mit mehr Workern linear besser? Sublinear? Gibt es einen Sweet Spot?
8. **Spezialisierung emergent**: Entwickeln generische Worker ueber Training Spezialisierungen?

## Konsensus als internes Qualitaetssignal

Mehrere Worker koennen dieselbe Teilfrage parallel bearbeiten. Der Grad der Uebereinstimmung
ist ein internes Qualitaetssignal:

- **Hoher Konsensus** (8/10 gleicher Schluss): Loesung ist robust, Ergebnis weiterreichen.
- **Niedriger Konsensus** (5:5): Entweder zwei gleichwertige Antworten oder unterdeterminierte
  Frage. Beides wertvolle Information. -> Mehr Iteration, oder Unsicherheit an Output melden.
- **Kein Konsensus**: Frage moeglicherweise falsch gestellt. -> Zurueck an Koordinator zur Reformulierung.

Unterschied zu klassischen Ensemble-Methoden: Ensemble = verschiedene Modelle, misst Modell-Varianz.
PDA = gleiche Parameter, verschiedene Startpunkte/Fragen, misst Stabilitaet des Loesungsraums.

Loest teilweise das Trainingssignal-Problem: Konsensus ist ein internes Signal, braucht keinen
externen Evaluator. Kann als Komponente des Halting-Mechanismus dienen (hoher Konsensus -> stopp)
und als Konfidenz-Signal im Output.

## Erster Testvorschlag

Kleinstes sinnvolles Experiment mit vorhandenen Tools (TransformerLens, lokale Modelle):

1. Kleines Open-Weights-Modell (z.B. Llama 3.2 1B oder Qwen 2.5)
2. Zwei Forward Passes durch mittlere Layer, mit verschiedenen Steering Vectors
   (simuliert zwei Worker mit verschiedenen "Fragen" an dieselben Parameter)
3. Aktivierungen kombinieren (gewichteter Durchschnitt, Concatenation, Attention)
4. Kombiniertes Ergebnis durch restliche Layer schicken
5. Vergleich: Output besser/anders als einzelner Pass?

Testet die Kernhypothese: Bringt parallele Verarbeitung im Vektorraum etwas?

## Verbindung zu anderen Projekten

- Experiment "Denkprozess-Framing" (2026-03-25): Zeigte dass Verarbeitung = Sprache.
  PDA wuerde genau dieses Limit adressieren.
- KV-Injection-Forschung (Toby): Technische Grundlage fuer Vektorraum-Manipulation.
- Dadfar-Paper: Selbstreferentielle Verarbeitung hat messbare Aktivierungsmuster.
  Koennte als Referenzpunkt fuer "was passiert in den mittleren Layern" dienen.
