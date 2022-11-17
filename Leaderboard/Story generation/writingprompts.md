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
man is granted the thing he most desires . problem is , after a certain amount of time , he will forget every memory of having it .
```

Output

```
daniel 's hands shook as he counted the coins . his hands always shook . it was n't the biting cold of december in scotland , it was n't the exhaustion of having to talk to everyone that walked past , it was n't the sheer excitement of finally having enough . they just always shook . he dropped the last penny in his pocket and smiled . a warmth flooded his body , he felt better than he had in weeks . he would finally be granted the thing he most desired . people gave him a wide berth as he walked down the street to the newsagent . daniel smiled even wider . the world was now on his side , everyone wanted him to get there as fast as possible .
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

