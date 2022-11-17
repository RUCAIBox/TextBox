# WMT14 English-French

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/W14-3302.pdf)

Homepage: [Official](https://www.statmt.org/wmt14/index.html)

Translation dataset based on the data from statmt.org.

### Overview

| Dataset              | Num Train    | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------------------- | ------------ | --------- | -------- | ------------------- | ------------------- |
| WMT14 English-French | $40,837,246$ | $3,000$   | $3,003$  | $28.4$              | $24.9$              |

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

| Model                                                        | BLEU    | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ | -------------- |
| [Transformer+BT](https://arxiv.org/pdf/2008.07772v2.pdf)     | $46.4$  | [Official](https://github.com/namisan/exdeep-nmt)            |                |
| [Noisy back-translation](https://arxiv.org/pdf/1808.09381v2.pdf) | $45.6$  | [Official](https://github.com/facebookresearch/fairseq)      |                |
| [mRASP+Fine-Tune](https://arxiv.org/pdf/2010.03142v3.pdf)    | $44.3$  | [Official](https://github.com/linzehui/mRASP)                |                |
| [Transformer + R-Drop](https://arxiv.org/pdf/2106.14448v2.pdf) | $43.95$ | [Official](https://github.com/dropreg/R-Drop)                |                |
| [Admin](https://arxiv.org/pdf/2004.08249v2.pdf)              | $43.8$  | [Official](https://github.com/LiyuanLucasLiu/Transforemr-Clinic) |                |
| [T5](https://arxiv.org/pdf/1910.10683v3.pdf)                 | $43.4$  | [Official](https://github.com/google-research/text-to-text-transfer-transformer) |                |
| [TaLK Convolutions](https://arxiv.org/pdf/2002.03184v2.pdf)  | $43.2$  | [Official](https://github.com/lioutasb/TaLKConvolutions)     |                |
| [OmniNetP](https://arxiv.org/pdf/2103.01075v1.pdf)           | $42.6$  |                                                              |                |
| [T2R + Pretrain](https://arxiv.org/pdf/2103.13076v2.pdf)     | $42.1$  |                                                              |                |
| [mBART](https://arxiv.org/pdf/2001.08210.pdf)                | $41.0$  | [Official](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart) |                |
| [ResMLP-6](https://arxiv.org/pdf/2105.03404v2.pdf)           | $40.3$  | [Official](https://github.com/facebookresearch/deit)         |                |
| [Rfa-Gate-arccos](https://arxiv.org/pdf/2103.02143v2.pdf)    | $39.2$  |                                                              |                |
| [FLAN 137B zero-shot](https://arxiv.org/pdf/2109.01652v5.pdf) | $34$    | [Official](https://github.com/google-research/flan)          |                |

#### French $\rarr$ English

Descending order by BLEU.

| Model                                                        | BLEU   | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ | -------------- |
| [GPT-3 175B](https://arxiv.org/pdf/2005.14165v4.pdf)         | $39.2$ | [Official](https://github.com/openai/gpt-3)                  |                |
| [MASS](https://arxiv.org/pdf/1905.02450v5.pdf)               | $34.9$ | [Official](https://github.com/microsoft/MASS)                |                |
| [SMT + NMT](https://arxiv.org/pdf/1902.01313v2.pdf)          | $33.5$ | [Official](https://github.com/artetxem/monoses)              |                |
| [MLM pretraining for encoder and decoder](https://arxiv.org/pdf/1901.07291v1.pdf) | $33.3$ |                                                              |                |
| [PBSMT + NMT](https://arxiv.org/pdf/1804.07755v2.pdf)        | $27.7$ | [Official](https://github.com/facebookresearch/UnsupervisedMT) |                |
| [SMT](https://arxiv.org/pdf/1809.01272v1.pdf)                | $25.9$ | [Official](https://github.com/artetxem/monoses)              |                |

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