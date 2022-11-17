# WOZ

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/P17-1163.pdf)

The WoZ 2.0 dataset is a newer dialogue state tracking dataset whose evaluation is detached from the noisy output of speech recognition systems. Similar to DSTC2, it covers the restaurant search domain and has identical evaluation.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| WOZ     | $6,364$   | $1,260$   | -        | $47.0$              | $10.6$              |

### Data Sample

Input

```
"Belief state [X_SEP] I'm looking for an expensive restaurant in the south part of town."
```

Output

```
'[restaurant] price range expensive area south'
```

## LeaderBoard

Descending order by Joint.

| Model                                                        | Joint   | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ | -------------- |
| [AG-DST](https://arxiv.org/pdf/2110.15659v1.pdf)             | $91.37$ | [Official](https://github.com/PaddlePaddle/Knover/tree/develop/projects/AG-DST) |                |
| [Seq2Seq-DU-w/oSchema](https://arxiv.org/pdf/2011.09553v2.pdf) | $91.2$  | [Official](https://github.com/sweetalyssum/Seq2Seq-DU)       |                |
| [T5(span)](https://arxiv.org/pdf/2108.13990v2.pdf)           | $91$    |                                                              |                |
| [StateNet](https://arxiv.org/pdf/1810.09587v1.pdf)           | $88.9$  | [Official](https://github.com/renll/StateNet)                |                |
| [G-SAT](https://arxiv.org/pdf/1910.09942v1.pdf)              | $88.7$  | [Official](https://github.com/vevake/GSAT)                   |                |
| [GCE](https://arxiv.org/pdf/1812.00899v1.pdf)                | $88.5$  | [Official](https://github.com/elnaaz/GCE-Model)              |                |

## Citation

```
@inproceedings{woz,
    title = "Neural Belief Tracker: Data-Driven Dialogue State Tracking",
    author = "Mrk{\v{s}}i{\'c}, Nikola  and
      {\'O} S{\'e}aghdha, Diarmuid  and
      Wen, Tsung-Hsien  and
      Thomson, Blaise  and
      Young, Steve",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1163",
    doi = "10.18653/v1/P17-1163",
    pages = "1777--1788",
}
```

