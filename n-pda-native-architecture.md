# Native Parallel Deliberation Architecture (n-PDA)

Status: Theoretisches Greenfield-Konzept (Grundlagenforschung)
Datum: 2026-03-27
Autoren: Toby, Claude, Gemini

## Kernidee

Statt bestehende Transformer-Modelle durch Inferenz-Eingriffe in einen parallelen
Deliberations-Modus zu zwingen (was oft zu Repraesentationsbruechen fuehrt), wird die
Architektur von Grund auf neu fuer paralleles Vektorraum-Reasoning konzipiert.
Das System ist nativ darauf trainiert, Perspektiven isoliert aufzubauen und iterativ
zu einem Konsensus zu verschmelzen.

Entstanden aus der Ueberlegung: Wenn man bei 0 anfangen wuerde, wie muesste die
Architektur aussehen, damit parallele Deliberation mathematisch sauber funktioniert?

## Verhaeltnis zum pragmatischen PDA

n-PDA ist kein Ersatz fuer PDA, sondern eine Denkuebung die zeigt, welche
mathematischen Eigenschaften fuer parallele Deliberation noetig sind:
- Orthogonalitaet (keine destruktive Interferenz)
- Differenzierbarer Austausch (nicht starrer Merge)
- Endogene Konvergenz (kein externer Halting-Mechanismus)

Erkenntnisse aus n-PDA fliessen in PDA-Experimente zurueck, z.B. orthogonale
Projektion als Merge-Preprocessing. Umgekehrt koennen PDA-Ergebnisse zeigen,
welche n-PDA-Annahmen empirisch tragen.

## Die drei Architektur-Saeulen

### 1. Orthogonale Perspektiv-Raeume (Das Fundament)

Worker teilen sich nicht denselben dichten Latenzraum. Verschiedene Perspektiven
werden durch Regularisierung beim Training in mathematisch orthogonale (rechtwinklige)
Unterraeume gezwungen.

Vorteil: Keine destruktive Interferenz beim Zusammenfuehren. Die Worker koennen
sich nicht versehentlich gegenseitig ueberschreiben oder ausloeschen.

Einschraenkung: Orthogonalitaet in hochdimensionalen Raeumen ist billig — in einem
4096-dimensionalen Raum passen tausende fast-orthogonale Vektoren. Die Frage ist
nicht ob die Unterraeume orthogonal SEIN koennen, sondern ob orthogonale Unterraeume
auch semantisch sinnvolle Perspektiven kodieren. Das ist eine empirische Frage,
die ueber Probing-Experimente geprueft werden kann (s. Simulations-Roadmap).

### 2. Native Cross-Attention (Der Deliberationsraum)

Der Austausch ist kein starrer Mittelwert-Merge am Ende eines Layers.
Stattdessen nutzen die Worker kontinuierliche Cross-Attention zwischen
ihren orthogonalen Raeumen.

Worker A "liest" differenzierbar in den Repraesentationen von Worker B und
zieht sich exakt die Merkmale, die seine Perspektive produktiv ergaenzen.

### 3. Deep Equilibrium Motor (Die Iteration)

Das Iterieren und das Halting-Kriterium werden nicht extern gesteuert.
Das Netzwerk wird als Deep Equilibrium Model (DEQ) formuliert.

Der Forward-Pass ist eine endogene Schleife, die iteriert, bis das System
einen mathematischen Fixpunkt erreicht (Delta x < epsilon). Der Konsensus
ist damit intrinsisch in den Forward-Pass eingebaut.

Vorteil: Kein externes Halting, keine willkuerlichen Schwellwerte.
Nachteil: DEQs sind notorisch schwer zu trainieren (instabile Gradienten).

## Architektur-Skizze

```
Input (Text)
    |
[Embedding & Orthogonale Projektion in n Unterraeume]
    |
    +---> [ DEQ Loop Start ] <---------------------+
    |                                              |
    |   [Worker 1] <--- Cross-Attention ---> [Worker n]
    |   (Sub-Raum 1)                         (Sub-Raum n)
    |                                              |
    +---> [ Konvergenz-Check: Divergenz < e ? ] ---+
                  |                      (Nein: Naechste Iteration)
                (Ja)
                  |
[ Konsolidierter Vektorraum ]
                  |
[ Unembedding-Layer ]
                  |
Output (Text)
```

## Offene Forschungsfragen

### Trainingsstabilitaet
DEQs sind schwer zu trainieren. Die Kombination mit orthogonaler
Regularisierung fuer die Unterraeume potenziert die Herausforderung.
Moegliche Mitigation: Jacobian-Free Backpropagation, Phantom Gradients.

### Loss-Funktion
Wie formuliert man ein Trainingssignal, das gleichzeitig:
- Task-Accuracy (Sprachverstaendnis/Generierung)
- Orthogonalitaet der Perspektiv-Raeume
- Schnelle DEQ-Konvergenz
belohnt?

Ansatz: Multi-Objective Loss mit Gewichtung, oder curriculares Training
(erst Orthogonalitaet, dann Task, dann Konvergenz-Speed).

### Skalierung
Kann diese Architektur auf die Groesse heutiger LLMs skaliert werden?
DEQ-Modelle sind bisher nur in relativ kleinen Varianten erfolgreich.

## Moegliche dritte Saeule: LoRA-Ensemble als gelernte Perspektiven

Zwischen PDA (Steering Vectors, handgecrafted) und n-PDA (orthogonale Raeume,
from scratch trainiert) gibt es einen Mittelweg:

Mehrere LoRA-Adapter auf dasselbe Basismodell trainieren, mit verschiedenen
Zielen oder Daten. Jeder LoRA IST eine gelernte Perspektive — nicht von Hand
gebaut, aber auch nicht so teuer wie ein ganzes Modell from scratch.

Vorteile:
- Sofort machbar mit bestehenden Tools (PEFT, HuggingFace)
- Perspektiven sind gelernt statt handgecrafted
- Bestehende Forschung zu LoRA-Merging (TIES, DARE, SLERP) direkt anwendbar
- Zwei Merge-Ebenen testbar: Weight-Merge (LoRAs zusammenfuehren, dann forwarden)
  vs. Aktivierungs-Merge (mehrere Forward Passes mit verschiedenen LoRAs)

Koennte als pragmatischer Einstieg in gelernte Perspektiven dienen, bevor
n-PDA-Training ueberhaupt machbar ist.

## Verwandte Dokumente

- PDA (Parallel Deliberation Architecture): Pragmatischer Ansatz mit bestehenden Modellen
- Simulations-Roadmap: Stufenplan zur Validierung der mathematischen Grundlagen
