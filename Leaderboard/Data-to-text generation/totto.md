# ToTTo

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/2004.14373)

Repository: [Official](https://github.com/google-research-datasets/ToTTo)

It is an open-domain English table-to-text dataset with over 120,000 training examples that proposes a controlled generation task: given a Wikipedia table and a set of highlighted table cells, produce a one-sentence description. 

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| ToTTo   | $120,761$ | $7,700$   | $7,700$  | $32.5$              | $14.8$              |

### Data Sample

Input

```
'page title | List of Governors of South Carolina [SEP] section title | Governors under the Constitution of 1868 [X_SEP] # | 76 [SEP] Governor | Daniel Henry Chamberlain [SEP] Took Office | December 1 , 1874'
```

Output

```
'Daniel Henry Chamberlain was the 76th Governor of South Carolina from 1874.', 'Daniel Henry Chamberlain was the 76th Governor of South Carolina, beginning in 1874.', 'Daniel Henry Chamberlain was the 76th Governor of South Carolina who took office in 1874.'
```

## LeaderBoard

Descending order by BLEU.

| Model                                                 | BLEU   | PARENT | Repository                                                   | Generated Text |
| ----------------------------------------------------- | ------ | ------ | ------------------------------------------------------------ | -------------- |
| [T5-3B](https://arxiv.org/pdf/2005.10433v3.pdf)       | $49.5$ | $58.4$ | [Official](https://github.com/google-research-datasets/ToTTo) |                |
| [PlanGen](https://arxiv.org/abs/2108.13740)           | $49.2$ | $58.7$ |                                                              |                |
| [CoNT](https://arxiv.org/abs/2205.14690v2)            | $49.1$ | $58.9$ | [Official](https://github.com/Shark-NLP/CoNT)                |                |
| [LATTICE](https://arxiv.org/pdf/2205.03972v1.pdf)     | $48.4$ | $58.1$ | [Official](https://github.com/luka-group/lattice)            |                |
| [Supervised+NLPO](https://arxiv.org/abs/2210.01241)   | $47.4$ | $59.6$ | [Official](https://github.com/allenai/rl4lms)                |                |
| [BERT-to-BERT](https://arxiv.org/abs/2004.14373)      | $44.0$ | $52.6$ |                                                              |                |
| [Pointer-Generator](https://arxiv.org/abs/2004.14373) | $41.6$ | $51.6$ |                                                              |                |

## Citation

```
 @inproceedings{totto2,
    title = "{ToTTo}: A Controlled Table-To-Text Generation Dataset",
    author = "Parikh, Ankur  and
      Wang, Xuezhi  and
      Gehrmann, Sebastian  and
      Faruqui, Manaal  and
      Dhingra, Bhuwan  and
      Yang, Diyi  and
      Das, Dipanjan",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.89",
    doi = "10.18653/v1/2020.emnlp-main.89",
    pages = "1173--1186",
}
```