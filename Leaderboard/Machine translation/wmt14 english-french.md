# WMT14 English-French

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/W14-3302.pdf)

Homepage: [Official](https://www.statmt.org/wmt14/index.html)

Translation dataset based on the data from statmt.org.

### Overview

| Dataset              | Num Train    | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------------------- | ------------ | --------- | -------- | ------------------- | ------------------- |
| WMT14 English-French | $40,836,876$ | $3,000$   | $3,003$  |                     |                     |

### Data Sample

English

```
Resumption of the session
```

French

```
Reprise de la session
```

## LeaderBoard

#### English $\rarr$ French

Descending order by BLEU.

| Model                                                        | BLEU   | Repository                                              | Generated Text |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------- | -------------- |
| [Transformer+BT](https://arxiv.org/pdf/2008.07772v2.pdf)     | $46.4$ | [Official](https://github.com/namisan/exdeep-nmt)       |                |
| [Noisy back-translation](https://arxiv.org/pdf/1808.09381v2.pdf) | $45.6$ | [Official](https://github.com/facebookresearch/fairseq) |                |
| [mRASP+Fine-Tune](https://arxiv.org/pdf/2010.03142v3.pdf)    | $44.3$ | [Official](https://github.com/linzehui/mRASP)           |                |

#### French $\rarr$ English

Descending order by BLEU.

| Model                                                | BLEU   | Repository                                      | Generated Text |
| ---------------------------------------------------- | ------ | ----------------------------------------------- | -------------- |
| [GPT-3 175B](https://arxiv.org/pdf/2005.14165v4.pdf) | $39.2$ | [Official](https://github.com/openai/gpt-3)     |                |
| [MASS](https://arxiv.org/pdf/1905.02450v5.pdf)       | $34.9$ | [Official](https://github.com/microsoft/MASS)   |                |
| [SMT + NMT](https://arxiv.org/pdf/1902.01313v2.pdf)  | $33.5$ | [Official](https://github.com/artetxem/monoses) |                |

## Citation

```
 @InProceedings{bojar-EtAl:2014:W14-33,
  author    = {Bojar, Ondrej  and  Buck, Christian  and  Federmann, Christian  and  Haddow, Barry  and  Koehn, Philipp  and  Leveling, Johannes  and  Monz, Christof  and  Pecina, Pavel  and  Post, Matt  and  Saint-Amand, Herve  and  Soricut, Radu  and  Specia, Lucia  and  Tamchyna, Ale{s}},
  title     = {Findings of the 2014 Workshop on Statistical Machine Translation},
  booktitle = {Proceedings of the Ninth Workshop on Statistical Machine Translation},
  month     = {June},
  year      = {2014},
  address   = {Baltimore, Maryland, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {12--58},
  url       = {http://www.aclweb.org/anthology/W/W14/W14-3302}
}
```