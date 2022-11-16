# MSQG

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/2011.11928)

Repository: [Official](https://github.com/microsoft/glge)

MSQG: MicroSoft Question Generation (MSQG) is a new challenge dataset, the questions in this dataset are freely edited by daily users. For MSQG, authors collect 220K passages from a real world search engine. Each passage contains a highlight span and a related query, we regard the queries as questions in this dataset.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| MSQG    | $198,058$ | $11,008$  | -        | $48.1$              | $3.7$               |

### Data Sample

Input

```
'causative medications [X_SEP] Genetic susceptibility and HLA associations have been found for several causative drugs. The most common drugs to cause this reaction are a number of anti-epilepsy drugs (particularly carbamazepine, phenobarbital and phenytoin), the anti-gout drug, allopurinol, olanzapine, and the sulphonamide group of antibiotics.'
```

Output

```
'anti-epilepsy drugs (particularly carbamazepine, phenobarbital and phenytoin), the anti-gout drug, allopurinol, olanzapine, and the sulphonamide group of antibiotics'
```

## LeaderBoard

#### Easy

Descending order by ROUGE-L.

| Model                                           | ROUGE-L | BLEU-4 | METEOR | Repository | Generated Text |
| ----------------------------------------------- | ------- | ------ | ------ | ---------- | -------------- |
| [MASS](https://arxiv.org/abs/2011.11928)        | $38.9$  | $9.5$  | $23.5$ |            |                |
| [BART](https://arxiv.org/abs/2011.11928)        | $38.8$  | $9.2$  | $24.3$ |            |                |
| [ProphetNet](https://arxiv.org/abs/2011.11928)  | $38.3$  | $9.6$  | $23.3$ |            |                |
| [Transformer](https://arxiv.org/abs/2011.11928) | $29.3$  | $5.1$  | $16.6$ |            |                |
| [LSTM](https://arxiv.org/abs/2011.11928)        | $25.3$  | $3.5$  | $14.1$ |            |                |

#### Medium

Descending order by ROUGE-L.

| Model                                           | ROUGE-L | BLEU-4 | METEOR | Repository | Generated Text |
| ----------------------------------------------- | ------- | ------ | ------ | ---------- | -------------- |
| [ProphetNet](https://arxiv.org/abs/2011.11928)  | $39.3$  | $10.0$ | $23.7$ |            |                |
| [BART](https://arxiv.org/abs/2011.11928)        | $38.4$  | $9.5$  | $24.0$ |            |                |
| [MASS](https://arxiv.org/abs/2011.11928)        | $38.3$  | $9.9$  | $22.7$ |            |                |
| [Transformer](https://arxiv.org/abs/2011.11928) | $27.0$  | $4.2$  | $15.0$ |            |                |
| [LSTM](https://arxiv.org/abs/2011.11928)        | $18.6$  | $1.7$  | $9.5$  |            |                |

#### Hard

Descending order by ROUGE-L.

| Model                                           | ROUGE-L | BLEU-4 | METEOR | Repository | Generated Text |
| ----------------------------------------------- | ------- | ------ | ------ | ---------- | -------------- |
| [ProphetNet](https://arxiv.org/abs/2011.11928)  | $34.0$  | $8.2$  | $19.0$ |            |                |
| [BART](https://arxiv.org/abs/2011.11928)        | $35.1$  | $8.5$  | $20.7$ |            |                |
| [MASS](https://arxiv.org/abs/2011.11928)        | $34.3$  | $8.0$  | $19.0$ |            |                |
| [Transformer](https://arxiv.org/abs/2011.11928) | $17.0$  | $1.3$  | $8.2$  |            |                |
| [LSTM](https://arxiv.org/abs/2011.11928)        | $12.5$  | $0.4$  | $5.0$  |            |                |

## Citation

```
@inproceedings{liu-etal-2021-glge,
    title = "{GLGE}: A New General Language Generation Evaluation Benchmark",
    author = "Liu, Dayiheng  and
      Yan, Yu  and
      Gong, Yeyun  and
      Qi, Weizhen  and
      Zhang, Hang  and
      Jiao, Jian  and
      Chen, Weizhu  and
      Fu, Jie  and
      Shou, Linjun  and
      Gong, Ming  and
      Wang, Pengcheng  and
      Chen, Jiusheng  and
      Jiang, Daxin  and
      Lv, Jiancheng  and
      Zhang, Ruofei  and
      Wu, Winnie  and
      Zhou, Ming  and
      Duan, Nan",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.36",
    doi = "10.18653/v1/2021.findings-acl.36",
    pages = "408--420",
}
```

