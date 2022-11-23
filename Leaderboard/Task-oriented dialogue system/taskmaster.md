# TaskMaster

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/D19-1459.pdf)

Homepage: [Official](https://research.google/tools/datasets/taskmaster-1/)

Repository: [Official](https://github.com/google-research-datasets/Taskmaster)

Taskmaster-1 is a dialog dataset consisting of 13,215 task-based dialogs in English, including 5,507 spoken and 7,708 written dialogs created with two distinct procedures. Each conversation falls into one of six domains: ordering pizza, creating auto repair appointments, setting up ride service, ordering movie tickets, ordering coffee drinks and making restaurant reservations.

### Overview

| Dataset    | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ---------- | --------- | --------- | -------- | ------------------- | ------------------- |
| TaskMaster | $249,664$ | $20,680$  | -        | $95.6$              | $12.0$              |

### Data Sample

Input

```
'Belief state [X_SEP] Hi Jim, could you send my car in for a tune up? Its kind of sluggish.'
```

Output

```
'[auto_repair] appt reason sluggish'
```

## LeaderBoard

Descending order by BLEU.

| Model                                                   | BLEU   | Repository | Generated Text |
| ------------------------------------------------------- | ------ | ---------- | -------------- |
| [Transformer](https://aclanthology.org/D19-1459.pdf)    | $6.11$ |            |                |
| [LSTM-attention](https://aclanthology.org/D19-1459.pdf) | $5.12$ |            |                |
| [Convolution](https://aclanthology.org/D19-1459.pdf)    | $5.09$ |            |                |
| [LSTM](https://aclanthology.org/D19-1459.pdf)           | $4.45$ |            |                |
| [4-gram](https://aclanthology.org/D19-1459.pdf)         | $0.21$ |            |                |
| [3-gram](https://aclanthology.org/D19-1459.pdf)         | $0.20$ |            |                |

## Citation

```
 @inproceedings{taskmaster,
    title = "Taskmaster-1: Toward a Realistic and Diverse Dialog Dataset",
    author = "Byrne, Bill  and
      Krishnamoorthi, Karthik  and
      Sankar, Chinnadhurai  and
      Neelakantan, Arvind  and
      Goodrich, Ben  and
      Duckworth, Daniel  and
      Yavuz, Semih  and
      Dubey, Amit  and
      Kim, Kyu-Young  and
      Cedilnik, Andy",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1459",
    doi = "10.18653/v1/D19-1459",
    pages = "4516--4525",
}
```