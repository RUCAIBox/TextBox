# NarrativeQA

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1712.07040)

Repository: [Official](https://github.com/deepmind/narrativeqa)

NarrativeQA is an English-lanaguage dataset of stories and corresponding questions designed to test reading comprehension, especially on long documents.

### Overview

| Dataset     | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ----------- | --------- | --------- | -------- | ------------------- | ------------------- |
| NattativeQA | $65,494$  | $6,922$   | $21,114$ | $584.1$             | $4.2$               |

### Data Sample

Input

```
'THE ACTOR WEARING THE BLACK CLOAK [X_SEP] The play begins with three pages disputing over the black cloak usually worn by the actor who delivers the prologue. They draw lots for the cloak, and one of the losers, Anaides, starts telling the audience what happens in the play to come; the others try to suppress him, interrupting him and putting their hands over his mouth. Soon they are fighting over the cloak and criticizing the author and the spectators as well. In the play proper, the goddess Diana, also called Cynthia, has ordained a "solemn revels" in the valley of Gargaphie in Greece. The gods Cupid and Mercury appear, and they too start to argue. Mercury has awakened Echo, who weeps for Narcissus, and states that a drink from Narcissus\'s spring causes the drinkers to "Grow dotingly enamored of themselves." The courtiers and ladies assembled for the Cynthia\'s revels all drink from the spring. Asotus, a foolish spendthrift who longs to become a courtier and a master of fashion and manners, also drinks from the spring; emboldened by vanity and self-love, he challenges all comers to a competition of "court compliment." The competition is held, in four phases, and the courtiers are beaten. Two symbolic masques are performed within the play for the assembled revelers. At their conclusion, Cynthia (representing Queen Elizabeth) has the dancers unmask and shows that vices have masqueraded as virtues. She sentences them to make reparation and to purify themselves by bathing in the spring at Mount Helicon. The figure of Actaeon in the play may represent Robert Devereux, 2nd Earl of Essex, while Cynthia\'s lady in waiting Arete may be Lucy, Countess of Bedford, one of Elizabeth\'s ladies in waiting as well as Jonson\'s patroness. The play is notably rich in music, as is typical for the theatre of the boys\' companies, which originated as church choirs.'
```

Output

```
'WHO NORMALLY DELIVERS THE OPENING PROLOGUE IN THE PLAY?'
```

## LeaderBoard

Descending order by METRIC.

| Model | METRIC | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
|       |        |            |                |
|       |        |            |                |
|       |        |            |                |

## Citation

```
@article{kocisky-etal-2018-narrativeqa,
    title = "The {N}arrative{QA} Reading Comprehension Challenge",
    author = "Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
      Schwarz, Jonathan  and
      Blunsom, Phil  and
      Dyer, Chris  and
      Hermann, Karl Moritz  and
      Melis, G{\'a}bor  and
      Grefenstette, Edward",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "6",
    year = "2018",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q18-1023",
    doi = "10.1162/tacl_a_00023",
    pages = "317--328",
}
```

 