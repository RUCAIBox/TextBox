# DailyDialog

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/I17-1099.pdf)

Homepage: [Official](http://yanran.li/dailydialog)

DailyDialog is a high-quality multi-turn open-domain English dialog dataset. It contains 13,118 dialogues split into a training set with 11,118 dialogues and validation and test sets with 1000 dialogues each. On average there are around 8 speaker turns per dialogue with around 15 tokens per turn.

### Overview

| Dataset     | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ----------- | --------- | --------- | -------- | ------------------- | ------------------- |
| DailyDialog | $76,052$  | $7,069$   | $6,740$  | $72.5$              | $13.9$              |

### Data Sample

```
{
    "act": [2, 1, 1, 1, 1, 2, 3, 2, 3, 4],
    "dialog": "[\"Good afternoon . This is Michelle Li speaking , calling on behalf of IBA . Is Mr Meng available at all ? \", \" This is Mr Meng ...",
    "emotion": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```

## LeaderBoard

Descending order by BLEU-2.

| Model                                                      | BLEU-1  | BLEU-2 | BLEU-3 | BLEU-4 | Repository                                           | Generated Text |
| ---------------------------------------------------------- | ------- | ------ | ------ | ------ | ---------------------------------------------------- | -------------- |
| [PLATO](https://arxiv.org/pdf/1910.07931v3.pdf)            | $39.7$  | $31.1$ | -      | -      | [Official](https://github.com/PaddlePaddle/Research) |                |
| [Seq2Seq](https://arxiv.org/pdf/1910.07931v3.pdf)          | $33.6$  | $26.8$ | -      | -      |                                                      |                |
| [iVAE$_{\rm{MI}}$](https://arxiv.org/pdf/1910.07931v3.pdf) | $30.9$  | $24.9$ | -      | -      |                                                      |                |
| [AEM+Attention](https://arxiv.org/pdf/1808.08795v1.pdf)    | $14.17$ | $5.69$ | $3.78$ | $2.84$ | [Official](https://github.com/lancopku/AMM)          |                |

## Citation

```
 @inproceedings{li-etal-2017-dailydialog,
    title = "{D}aily{D}ialog: A Manually Labelled Multi-turn Dialogue Dataset",
    author = "Li, Yanran  and
      Su, Hui  and
      Shen, Xiaoyu  and
      Li, Wenjie  and
      Cao, Ziqiang  and
      Niu, Shuzi",
    booktitle = "Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = nov,
    year = "2017",
    address = "Taipei, Taiwan",
    publisher = "Asian Federation of Natural Language Processing",
    url = "https://aclanthology.org/I17-1099",
    pages = "986--995",
}
```