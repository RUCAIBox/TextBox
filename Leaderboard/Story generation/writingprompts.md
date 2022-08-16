# WritingPrompts

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/P18-1082.pdf)

Homepage: [Official](https://www.kaggle.com/datasets/ratthachat/writing-prompts)

WritingPrompts is a large dataset of 300K human-written stories paired with writing prompts from an online forum.

### Overview

| Dataset        | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------------- | --------- | --------- | -------- | ------------------- | ------------------- |
| WritingPrompts | $272,600$ | $15,620$  | $15,138$ | $28.4$              | $927.4$             |

### Data Sample

Input

```

```

Output

```

```

## LeaderBoard

Ascending order by Perplexity.

| Model                                                        | Perplexity | Repository | Generated Text |
| ------------------------------------------------------------ | ---------- | ---------- | -------------- |
| [Conv seq2seq + self-attention](https://aclanthology.org/P18-1082.pdf) | $36.56$    |            |                |
| [Conv seq2seq](https://aclanthology.org/P18-1082.pdf)        | $45.54$    |            |                |
| [LSTM seq2seq](https://aclanthology.org/P18-1082.pdf)        | $46.79$    |            |                |
| [GCNN + self-attention LM](https://aclanthology.org/P18-1082.pdf) | $51.18$    |            |                |
| [GCNN LM](https://aclanthology.org/P18-1082.pdf)             | $54.79$    |            |                |

## Citation

```
@inproceedings{fan-etal-2018-hierarchical,
    title = "Hierarchical Neural Story Generation",
    author = "Fan, Angela  and
      Lewis, Mike  and
      Dauphin, Yann",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-1082",
    doi = "10.18653/v1/P18-1082",
    pages = "889--898",
}
```

