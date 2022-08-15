# WikiBio

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/pdf/1603.07771v3.pdf)

It is a new dataset of biographies from Wikipedia.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| WikiBio | $582,659$ | $72,831$  | $72,831$ | $81.6$              | $26.1$              |

### Data Sample

Input

```

```

Output

```

```

## LeaderBoard

Descending order by BLEU.

| Model                                               | BLEU    | Repository                                              | Generated Text |
| --------------------------------------------------- | ------- | ------------------------------------------------------- | -------------- |
| [ MBD](https://arxiv.org/pdf/2102.02810v2.pdf)      | $41.56$ | [Official](https://github.com/KaijuML/dtt-multi-branch) |                |
| [Table NLM](https://arxiv.org/pdf/1603.07771v3.pdf) | $34.70$ |                                                         |                |

## Citation

```
 @inproceedings{wikibio,
    title = "Neural Text Generation from Structured Data with Application to the Biography Domain",
    author = "Lebret, R{\'e}mi  and
      Grangier, David  and
      Auli, Michael",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D16-1128",
    doi = "10.18653/v1/D16-1128",
    pages = "1203--1213",
}
```