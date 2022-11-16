# MuTual

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/2020.acl-main.130.pdf)

Repository: [Official](https://github.com/Nealcly/MuTual)

MuTual is a retrieval-based dataset for multi-turn dialogue reasoning, which is modified from Chinese high school English listening comprehension test data. It tests dialogue reasoning via next utterance prediction.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| MuTual  | $33,691$  | $4,090$   | $3,248$  | $53.6$              | $14.5$              |

### Data Sample

Input

```
"Good evening. For today's program, we have invited Sophie, a lady, who worked as a nurse during World War 2."
```

Output

```
'Good evening. At the beginning of World War 2, the government called on all its citizens 18 years old and over to help out. I started training as a nurse in November 1940. It was 2 months of being taught basic skills in the school of nursing.'
```

## LeaderBoard

Descending order by R@1.

| Model                                                        | R@1    | R@2    | MRR    | Repository | Generated Text |
| ------------------------------------------------------------ | ------ | ------ | ------ | ---------- | -------------- |
| [Human](https://aclanthology.org/2020.acl-main.130.pdf)      | $93.8$ | $97.1$ | $96.4$ |            |                |
| [RoBERTa](https://aclanthology.org/2020.acl-main.130.pdf)    | $71.3$ | $89.2$ | $83.6$ |            |                |
| [RoBERTa-MC](https://aclanthology.org/2020.acl-main.130.pdf) | $68.6$ | $88.7$ | $82.2$ |            |                |
| [BERT-MC](https://aclanthology.org/2020.acl-main.130.pdf)    | $66.7$ | $87.8$ | $81.0$ |            |                |
| [BERT](https://aclanthology.org/2020.acl-main.130.pdf)       | $64.8$ | $84.7$ | $79.5$ |            |                |
| [SMN](https://aclanthology.org/2020.acl-main.130.pdf)        | $29.9$ | $58.5$ | $59.5$ |            |                |
| [TF-IDF](https://aclanthology.org/2020.acl-main.130.pdf)     | $27.9$ | $53.6$ | $54.2$ |            |                |
| [Dual LSTM](https://aclanthology.org/2020.acl-main.130.pdf)  | $26.0$ | $49.1$ | $74.3$ |            |                |
| [DAM](https://aclanthology.org/2020.acl-main.130.pdf)        | $24.1$ | $46.5$ | $51.8$ |            |                |

## Citation

```
 @inproceedings{cui-etal-2020-mutual,
    title = "{M}u{T}ual: A Dataset for Multi-Turn Dialogue Reasoning",
    author = "Cui, Leyang  and
      Wu, Yu  and
      Liu, Shujie  and
      Zhang, Yue  and
      Zhou, Ming",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.130",
    doi = "10.18653/v1/2020.acl-main.130",
    pages = "1406--1416",
}
```