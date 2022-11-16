# WMT19 Czech-English

## Dataset

### Instruction

Homepage: [Official](http://www.statmt.org/wmt19/translation-task.html)

Translation dataset based on the data from statmt.org.

### Overview

| Dataset             | Num Train    | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------------------- | ------------ | --------- | -------- | ------------------- | ------------------- |
| WMT19 Czech-English | $20,246,548$ | $2,983$   | $2,983$  | $12.6$              | $13.8$              |

### Data Sample

English

```
Action taken on Parliament's resolutions: see Minutes
```

Czech

```
Následný postup na základě usnesení Parlamentu: viz zápis
```

## LeaderBoard

#### English $\rarr$ Czech

Descending order by BLEU.

| Model | BLEU | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
| [mBART](https://arxiv.org/pdf/2001.08210.pdf) | $18.0$ | [Official](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart) |                |

#### Czech $\rarr$ English

Descending order by BLEU.

| Model | BLEU | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
|       |        |            |                |
|       |        |            |                |
|       |        |            |                |

## Citation

```
@ONLINE {wmt19translate,
    author = "Wikimedia Foundation",
    title  = "ACL 2019 Fourth Conference on Machine Translation (WMT19), Shared Task: Machine Translation of News",
    url    = "http://www.statmt.org/wmt19/translation-task.html"
}
```