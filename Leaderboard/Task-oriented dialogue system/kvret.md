# KVRET

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/W17-5506.pdf)

Repository: [Official](https://nlp.stanford.edu/projects/kvret/)

The KVRET corpus introduced to evaluate Key-Value Retrieval Networks for Task-Oriented Dialogue is a multi-turn, multi-domain dialogue dataset of 3,031 dialogues that are grounded through underlying knowledge bases and span three distinct tasks in the in-car personal assistant space: calendar scheduling, weather information retrieval, and point-of-interest navigation.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| KVRET   | $14,136$  | $1,616$   | -        | $30.5$              | $9.3$               |

### Data Sample

Input

```
'Belief state [X_SEP] remind me to take my pills'
```

Output

```
'[car_assistant] event take pills'
```

## LeaderBoard

Descending order by Entity-F1.

| Model                                                       | Entity-F1 | BLEU    | Repository                                                   | Generated Text |
| ----------------------------------------------------------- | --------- | ------- | ------------------------------------------------------------ | -------------- |
| [T5-3B(UnifiedSKG)](https://arxiv.org/pdf/2201.05966v2.pdf) | $70.07$   |         | [Official](https://github.com/hkunlp/unifiedskg)             |                |
| [ COMET](https://arxiv.org/pdf/2010.05740v4.pdf)            | $63.6$    | $17.3$  |                                                              |                |
| [DF-Net](https://arxiv.org/pdf/2004.11019v3.pdf)            | $62.5$    | $15.2$  | [Official](https://github.com/LooperXX/DF-Net)               |                |
| [GLMP](https://arxiv.org/pdf/1901.04713v2.pdf)              | $59.97$   | $14.79$ | [Official](https://github.com/jasonwu0731/GLMP)              |                |
| [TTOS](https://aclanthology.org/2020.emnlp-main.281.pdf)    | $55.38$   | $17.35$ | [Official](https://github.com/siat-nlp/TTOS)                 |                |
| [KB-retriever](https://arxiv.org/pdf/1909.06762v2.pdf)      | $53.7$    | $13.9$  | [Official](https://github.com/yizhen20133868/Retriever-Dialogue) |                |
| [DSR](https://arxiv.org/pdf/1806.04441v1.pdf)               | $51.9$    | $12.7$  |                                                              |                |
| [KV Retrieval Net](https://arxiv.org/pdf/1705.05414v2.pdf)  | $48.0$    | $13.2$  |                                                              |                |
| [THPN](https://aclanthology.org/2021.dialdoc-1.3.pdf)       | $37.8$    | $12.8$  | [Official](https://github.com/wdimmy/THPN)                   |                |
| [MeM2Seq](https://arxiv.org/pdf/1804.08217v3.pdf)           | $33.4$    | $12.6$  | [Official](https://github.com/HLTCHKUST/Mem2Seq)             |                |

## Citation

```
@inproceedings{eric-etal-2017-key,
    title = "Key-Value Retrieval Networks for Task-Oriented Dialogue",
    author = "Eric, Mihail  and
      Krishnan, Lakshmi  and
      Charette, Francois  and
      Manning, Christopher D.",
    booktitle = "Proceedings of the 18th Annual {SIG}dial Meeting on Discourse and Dialogue",
    month = aug,
    year = "2017",
    address = {Saarbr{\"u}cken, Germany},
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-5506",
    doi = "10.18653/v1/W17-5506",
    pages = "37--49",
}
```

