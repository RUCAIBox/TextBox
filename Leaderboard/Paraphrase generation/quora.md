# Quora

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/2005.08417)

Repository: [Official](https://github.com/malllabiisc/SGCP)

The original Quora Question Pairs (QQP) dataset contains about 400K sentence pairs labeled positive if they are duplicates of each other and negative otherwise. The dataset is composed of about 150K positive and 250K negative pairs. We select those positive pairs which contain both sentences with a maximum token length of 30, leaving us with ~146K pairs.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| Quora   |           |           |          |                     |                     |

### Data Sample

Input

```

```

Output

```

```

## LeaderBoard

Descending order by BLEU.

| Model                                    | BLEU   | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | Repository                                      | Generated Text |
| ---------------------------------------- | ------ | ------ | ------- | ------- | ------- | ----------------------------------------------- | -------------- |
| [SGCP](https://arxiv.org/abs/2005.08417) | $38.0$ | $41.3$ | $68.1$  | $45.7$  | $70.2$  | [Official](https://github.com/malllabiisc/SGCP) |                |
| [CGEN](https://arxiv.org/abs/2005.08417) | $34.9$ | $37.4$ | $62.6$  | $42.7$  | $65.4$  |                                                 |                |
| [SCPN](https://arxiv.org/abs/2005.08417) | $15.6$ | $19.6$ | $40.6$  | $20.5$  | $44.6$  |                                                 |                |

## Citation

```
 @article{quora,
    title = "Syntax-Guided Controlled Generation of Paraphrases",
    author = "Kumar, Ashutosh  and
      Ahuja, Kabir  and
      Vadapalli, Raghuram  and
      Talukdar, Partha",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "8",
    year = "2020",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2020.tacl-1.22",
    doi = "10.1162/tacl_a_00318",
    pages = "329--345",
}
```