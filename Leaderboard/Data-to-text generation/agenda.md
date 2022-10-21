# AGENDA

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1904.02342)

Repository: [Official](https://github.com/rikdz/GraphWriter)

It's a large dataset of knowledge-graphs paired with scientific texts.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| AGENDA  | $38,720$  | $1,000$   | $1,000$  | $52.1$              | $141.2$             |

### Data Sample

Input

```
'Title | Hierarchical Semantic Classification : Word Sense Disambiguation with World Knowledge . [X_SEP] learning architecture | USED-FOR | lexical semantic classification problems'
```

Output

```
"we present a learning architecture for lexical semantic classification problems that supplements task-specific training data with background data encoding general '' world knowledge '' . the learning architecture compiles knowledge contained in a dictionary-ontology into additional training data , and integrates task-specific and background data through a novel hierarchical learning architecture . experiments on a word sense disambiguation task provide empirical evidence that this '' hierarchical learning architecture '' outperforms a state-of-the-art standard '' flat '' one ."
```

## LeaderBoard

Descending order by BLEU.

| Model                                                     | BLEU    | METEOR  | Repository                                            | Generated Text |
| --------------------------------------------------------- | ------- | ------- | ----------------------------------------------------- | -------------- |
| [BART-large+STA](https://arxiv.org/pdf/2007.08426v3.pdf)  | $25.66$ |         | [Official](https://github.com/UKPLab/plms-graph2text) |                |
| [BART-large](https://arxiv.org/pdf/2007.08426v3.pdf)      | $23.65$ |         |                                                       |                |
| [Writer-Reviewer](https://arxiv.org/pdf/2101.00916v1.pdf) | $19.60$ |         | [Official](https://github.com/menggehe/DRAW)          |                |
| [CGE-LW](https://arxiv.org/pdf/2001.11003v2.pdf)          | $18.01$ |         | [Official](https://github.com/UKPLab/kg2text)         |                |
| [Graformer](https://arxiv.org/pdf/2001.11003v2.pdf)       | $17.80$ |         |                                                       |                |
| [GraphWriter](https://arxiv.org/abs/1904.02342)           | $14.3$  | $18.8$  | [Official](https://github.com/rikdz/GraphWriter)      |                |
| [GAT](https://arxiv.org/abs/1904.02342)                   | $12.2$  | $17.2$  |                                                       |                |
| [EntityWriter](https://arxiv.org/abs/1904.02342)          | $10.38$ | $16.53$ |                                                       |                |
| [Rewriter](https://arxiv.org/abs/1904.02342)              | $1.05$  | $8.38$  |                                                       |                |

## Citation

```
 @inproceedings{agenda,
    title = "{T}ext {G}eneration from {K}nowledge {G}raphs with {G}raph {T}ransformers",
    author = "Koncel-Kedziorski, Rik  and
      Bekal, Dhanush  and
      Luan, Yi  and
      Lapata, Mirella  and
      Hajishirzi, Hannaneh",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1238",
    doi = "10.18653/v1/N19-1238",
    pages = "2284--2293",
}
```