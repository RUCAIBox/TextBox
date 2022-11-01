# CommonGen

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1911.03705)

Repository: [Official](https://inklab.usc.edu/CommonGen/)

CommonGen is a constrained text generation task, associated with a benchmark dataset, to explicitly test machines for the ability of generative commonsense reasoning. Given a set of common concepts; the task is to generate a coherent sentence describing an everyday scenario using these concepts.

CommonGen is challenging because it inherently requires 1) relational reasoning using background commonsense knowledge, and 2) compositional generalization ability to work on unseen concept combinations. Our dataset, constructed through a combination of crowd-sourcing from AMT and existing caption corpora, consists of 30k concept-sets and 50k sentences in total.

### Overview

| Dataset   | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| --------- | --------- | --------- | -------- | ------------------- | ------------------- |
| CommonGen | $67,389$  | $4,018$   | $1,497$  | 5.5                 | 11.6                |

### Data Sample

Input
```
[ "ski", "mountain", "skier" ]
```
Output
```
Skier skis down the mountain
```
## LeaderBoard

Descending order by SPICE.

| Model                                                        | BLUE-4   | CIDEr    | SPICE    | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | -------- | -------- | -------- | ------------------------------------------------------------ | -------------- |
| [KFCNet](https://arxiv.org/abs/2109.06704)                   | $43.619$ | $18.845$ | $33.911$ |                                                              |                |
| [KGR$^4$](https://arxiv.org/abs/2112.08266)                  | $42.818$ | $18.423$ | $33.564$ | [Official](https://github.com/DeepLearnXMU/KGR-4)            |                |
| [PU-GEN + T5-large](https://www.sciencedirect.com/science/article/abs/pii/S0950705122009546?via%3Dihub) | $38.233$ | $18.036$ | $31.682$ |                                                              |                |
| [Imagine-and-Verbalize](https://openreview.net/forum?id=Oh1r2wApbPv) | $40.565$ | $17.716$ | $31.291$ | [Official](https://github.com/wangpf3/imagine-and-verbalize) |                |
| [RE-T5](https://aclanthology.org/2021.findings-acl.269/)     | $40.863$ | $17.663$ | $31.079$ | [Official](https://github.com/HanNight/RE-T5)                |                |
| [VisCTG](https://arxiv.org/pdf/2109.03892.pdf)               | $36.939$ | $17.199$ | $29.973$ | [Official](https://github.com/styfeng/VisCTG)                |                |
| [SAPPHIRE](https://aclanthology.org/2021.inlg-1.21.pdf)      | $37.119$ | $16.901$ | $29.751$ | [Official](https://github.com/styfeng/sapphire)              |                |
| [KG-BART](https://arxiv.org/pdf/2009.12677.pdf)              | $33.867$ | $16.927$ | $29.634$ | [Official](https://github.com/yeliu918/KG-BART)              |                |
| [EKI-BART](https://arxiv.org/pdf/2012.00366.pdf)             | $35.945$ | $16.999$ | $29.583$ |                                                              |                |
| [T5-Large](https://arxiv.org/abs/1910.10683)                 | $31.962$ | $15.128$ | $28.855$ |                                                              |                |
| [BART](https://arxiv.org/abs/1910.13461)                     | $31.827$ | $13.976$ | $27.995$ |                                                              |                |
| [UniLM](https://arxiv.org/abs/1905.03197v3)                  | $30.616$ | $14.889$ | $27.429$ |                                                              |                |
| [BERT-Gen]()                                                 | $23.468$ | $12.606$ | $24.822$ | [Official](https://github.com/microsoft/unilm/tree/master/s2s-ft) |                |
| [GPT-2](https://github.com/openai/gpt-2)                     | $26.833$ | $12.187$ | $23.567$ |                                                              |                |

## Citation

```
@inproceedings{lin-etal-2020-commongen,
    title = "{C}ommon{G}en: A Constrained Text Generation Challenge for Generative Commonsense Reasoning",
    author = "Lin, Bill Yuchen  and
      Zhou, Wangchunshu  and
      Shen, Ming  and
      Zhou, Pei  and
      Bhagavatula, Chandra  and
      Choi, Yejin  and
      Ren, Xiang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.165",
    doi = "10.18653/v1/2020.findings-emnlp.165",
    pages = "1823--1840",
} 
```