# WMT19 Chinese-English

## Dataset

### Instruction

Homepage: [Official](http://www.statmt.org/wmt19/translation-task.html)

Translation dataset based on the data from statmt.org.

### Overview

| Dataset               | Num Train    | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| --------------------- | ------------ | --------- | -------- | ------------------- | ------------------- |
| WMT19 Chinese-English | $25,986,436$ | $3,981$   | $3,981$  | $39.6$              | $20.7$              |

### Data Sample

English

```
PARIS – As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.
```

Chinese

```
巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。
```

## LeaderBoard

#### English $\rarr$ Chinese

Descending order by BLEU.

| Model                                         | BLEU   | Repository                                                   | Generated Text |
| --------------------------------------------- | ------ | ------------------------------------------------------------ | -------------- |
| [mBART](https://arxiv.org/pdf/2001.08210.pdf) | $33.3$ | [Official](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart) |                |

#### Chinese $\rarr$ English

Descending order by BLEU.

| Model | BLEU | Repository | Generated Text |
| ----- | ---- | ---------- | -------------- |
|       |      |            |                |
|       |      |            |                |
|       |      |            |                |

## Citation

```
@ONLINE {wmt19translate,
    author = "Wikimedia Foundation",
    title  = "ACL 2019 Fourth Conference on Machine Translation (WMT19), Shared Task: Machine Translation of News",
    url    = "http://www.statmt.org/wmt19/translation-task.html"
}
```