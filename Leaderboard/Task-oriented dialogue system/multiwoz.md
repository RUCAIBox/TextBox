# MultiWOZ

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/D18-1547.pdf)

Repository: [Official](https://github.com/budzianowski/multiwoz)

The Multi-domain Wizard-of-Oz (MultiWOZ) dataset is a large-scale human-human conversational corpus spanning over seven domains, containing 8438 multi-turn dialogues, with each dialogue averaging 14 turns. Different from existing standard datasets like WOZ and DSTC2, which contain less than 10 slots and only a few hundred values, MultiWOZ has 30 (domain, slot) pairs and over 4,500 possible values. The dialogues span seven domains: restaurant, hotel, attraction, taxi, train, hospital and police.

### Overview

| Dataset  | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | -------- | ------------------- | ------------------- |
| MultiWOZ | $170,220$ | $22,074$  | $22,116$ | $128.3$             | $11.3$              |

### Data Sample

Input

```
Belief state [X_SEP] i need to book a hotel in the east that has 4 stars .
```

Output

```
[hotel] area east stars 4
```

## LeaderBoard

Descending order by BLEU.

| Model                                                        | BLEU   | Repository                                              | Generated Text |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------- | -------------- |
| [ Noisy Channel Model](https://arxiv.org/pdf/2103.10518v1.pdf) | $20.6$ |                                                         |                |
| [GALAXY](https://arxiv.org/pdf/2111.14592v8.pdf)             | $20.5$ | [Official](https://github.com/siat-nlp/galaxy)          |                |
| [ DAMD](https://arxiv.org/pdf/1911.10484v2.pdf)              | $18.6$ | [Official](https://gitlab.com/ucdavisnlp/damd-multiwoz) |                |
| [AuGPT](https://arxiv.org/pdf/2102.05126v3.pdf)              | $17.2$ | [Official](https://github.com/ufal/augpt)               |                |
| [SOLOIST](https://arxiv.org/pdf/2005.05298v4.pdf)            | $16.5$ |                                                         |                |
| [SimpleTOD](https://arxiv.org/pdf/2005.00796v4.pdf)          | $15.0$ | [Official](https://github.com/salesforce/simpletod)     |                |

## Citation

```
@inproceedings{budzianowski-etal-2018-multiwoz,
    title = "{M}ulti{WOZ} - A Large-Scale Multi-Domain {W}izard-of-{O}z Dataset for Task-Oriented Dialogue Modelling",
    author = "Budzianowski, Pawe{\l}  and
      Wen, Tsung-Hsien  and
      Tseng, Bo-Hsiang  and
      Casanueva, I{\~n}igo  and
      Ultes, Stefan  and
      Ramadan, Osman  and
      Ga{\v{s}}i{\'c}, Milica",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1547",
    doi = "10.18653/v1/D18-1547",
    pages = "5016--5026",
}
```

