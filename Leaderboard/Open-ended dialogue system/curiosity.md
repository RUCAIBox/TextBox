# Curiosity

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/2020.emnlp-main.655.pdf)

Homepage: [Official](http://curiosity.pedro.ai/)

The Curiosity dataset consists of 14K dialogs (with 181K utterances) with fine-grained knowledge groundings, dialog act annotations, and other auxiliary annotation. In this dataset users and virtual assistants converse about geographic topics like geopolitical entities and locations. This dataset is annotated with pre-existing user knowledge, message-level dialog acts, grounding to Wikipedia, and user reactions to messages.

### Overview

| Dataset   | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| --------- | --------- | --------- | -------- | ------------------- | ------------------- |
| Curiosity | $64,930$  | $8,539$   | $8,495$  | $144.4$             | $20.2$              |

### Data Sample

Input

```
"Topic | Etymology [SEP] Aspect | Culture, Etymology [SEP] Known Entities | Barack Obama, China, Chinese language [X_SEP] Hi! I would like very much to know about Vietnam's etymology please."
```

Output

```
'It is located in Southeast Asia, it has five countries and only one of them are communist countries.'
```

## LeaderBoard

Descending order by ROUGE-L.

| Model                                                        | ROUGE-L | BERTScore | Repository | Generated Text |
| ------------------------------------------------------------ | ------- | --------- | ---------- | -------------- |
| [ISEEQ-ERL](https://arxiv.org/pdf/2112.07622v1.pdf)          | $79$    | $66$      |            |                |
| [ProphetNet-FT SQUAD](https://arxiv.org/pdf/2112.07622v1.pdf) | $74$    | $85$      |            |                |
| [T5-FT CANARD](https://arxiv.org/pdf/2112.07622v1.pdf)       | $65$    | $79$      |            |                |

## Citation

```
@inproceedings{rodriguez-etal-2020-information,
    title = "Information Seeking in the Spirit of Learning: A Dataset for Conversational Curiosity",
    author = "Rodriguez, Pedro  and
      Crook, Paul  and
      Moon, Seungwhan  and
      Wang, Zhiguang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.655",
    doi = "10.18653/v1/2020.emnlp-main.655",
    pages = "8153--8172",
} 
```