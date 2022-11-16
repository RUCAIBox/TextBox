# WMT19 Russian-English

## Dataset

### Instruction

Homepage: [Official](http://www.statmt.org/wmt19/translation-task.html)

Translation dataset based on the data from statmt.org.

### Overview

| Dataset               | Num Train    | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| --------------------- | ------------ | --------- | -------- | ------------------- | ------------------- |
| WMT19 Russian-English | $38,492,126$ | $3,000$   | $3,000$  | $17.1$              | $19.0$              |

### Data Sample

English

```
Russian President Vladimir Putin has signed a law on the establishment of administrative liability for violating the deadline and procedures for payment of goods (works, services) as part of procurement for state and municipal needs.
```

Russian

```
резидент России Владимир Путин подписал закон о введении административной ответственности за нарушение срока и порядка оплаты товаров (работ, услуг) при осуществлении закупок для государственных и муниципальных нужд.
```

## LeaderBoard

#### English $\rarr$ Russian

Descending order by BLEU.

| Model                                         | BLEU   | Repository                                                   | Generated Text |
| --------------------------------------------- | ------ | ------------------------------------------------------------ | -------------- |
| [mBART](https://arxiv.org/pdf/2001.08210.pdf) | $31.3$ | [Official](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart) |                |

#### Russian $\rarr$ English

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

