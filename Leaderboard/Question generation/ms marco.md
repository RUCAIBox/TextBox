# MS MARCO

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1611.09268)

Homepage: [Official](https://microsoft.github.io/msmarco/)

The MS MARCO (Microsoft MAchine Reading Comprehension) is a collection of datasets focused on deep learning in search. The first dataset was a question answering dataset featuring 100,000 real Bing questions and a human generated answer. Over time the collection was extended with a 1,000,000 question dataset, a natural language generation dataset, a passage ranking dataset, keyphrase extraction dataset, crawling dataset, and a conversational search.

### Overview

| Dataset  | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | -------- | ------------------- | ------------------- |
| MS MARCO | $681,445$ | $77,580$  | -        | $68.7$              | $13.3$              |

### Data Sample

Input

```
"A corporation is a company or group of people authorized to act as a single entity and recognized as such in law. [X_SEP] McDonald's Corporation is one of the most recognizable corporations in the world. A corporation is a company or group of people authorized to act as a single entity (legally a person) and recognized as such in law. Early incorporated entities were established by charter (i.e. by an ad hoc act granted by a monarch or passed by a parliament or legislature)."
```

Output

```
'. what is a corporation?'
```

## LeaderBoard

Descending order by METRICL.

| Model | METRIC | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
|       |        |            |                |
|       |        |            |                |
|       |        |            |                |

## Citation

```
@inproceedings{marco,
  author    = {Tri Nguyen and
               Mir Rosenberg and
               Xia Song and
               Jianfeng Gao and
               Saurabh Tiwary and
               Rangan Majumder and
               Li Deng},
  title     = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  booktitle = {CoCo@NIPS},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {1773},
  publisher = {CEUR-WS.org},
  url={http://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper9.pdf},
  year      = {2016}
}
```

 