# WMT16 Romanian-English

## Dataset

### Instruction

Paper: [Paper](http://www.aclweb.org/anthology/W/W16/W16-2301)

Homepage: [Official](https://www.statmt.org/wmt16/index.html)

Translate dataset based on the data from statmt.org.

### Overview

| Dataset                | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ---------------------- | --------- | --------- | -------- | ------------------- | ------------------- |
| WMT16 Romanian-English | $610,320$ | $1,999$   | $1,999$  | $23.4$              | $23.1$              |

### Data Sample

English

```
Membership of Parliament: see Minutes
```

Romanian

```
Componenţa Parlamentului: a se vedea procesul-verbal
```

## LeaderBoard

#### English $\rarr$ Romanian

Descending order by BLEU.

| Model                                                        | BLEU    | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ | -------------- |
| [mBART](https://arxiv.org/pdf/2001.08210.pdf)                | $37.7$  | [Official](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart) |                |
| [DeLighT](https://arxiv.org/pdf/2008.00623v2.pdf)            | $34.7$  | [Official](https://github.com/sacmehta/delight)              |                |
| [CMLM+LAT+4 iterations](https://arxiv.org/pdf/2011.06132v1.pdf) | $32.87$ | [Official](https://github.com/shawnkx/NAT-with-Local-AT)     |                |
| [FlowSeq-large](https://arxiv.org/pdf/1909.02480v3.pdf)      | $32.35$ | [Official](https://github.com/XuezheMax/flowseq)             |                |
| [CMLM+LAT+4 iterations](https://arxiv.org/pdf/2011.06132v1.pdf) | $30.74$ | [Official](https://github.com/shawnkx/NAT-with-Local-AT)     |                |
| [ConvS2S BPE40k](https://arxiv.org/pdf/1705.03122v3.pdf)     | $29.9$  | [Official](https://github.com/facebookresearch/fairseq)      |                |
| [FLAN 137B zero-shot](https://arxiv.org/pdf/2109.01652v5.pdf) | $18.4$  | [Official](https://github.com/google-research/flan)          |                |

#### Romanian $\rarr$ English

Descending order by BLEU.

| Model                                                        | BLEU    | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ | -------------- |
| [fast-noisy-channel-modeling](https://arxiv.org/pdf/2011.07164v1.pdf) | $40.3$  | [Official](https://github.com/facebookresearch/fairseq/tree/main/examples/fast_noisy_channel) |                |
| [mBART](https://arxiv.org/pdf/2001.08210.pdf)                | $37.8$  | [Official](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart) |                |
| [FLAN 137B zero-shot](https://arxiv.org/pdf/2109.01652v5.pdf) | $36.7$  | [Official](https://github.com/google-research/flan)          |                |
| [MLM pretraining](https://arxiv.org/pdf/1901.07291v1.pdf)    | $35.3$  |                                                              |                |
| [Levenshtein Transformer](https://arxiv.org/pdf/1905.11006v2.pdf) | $33.26$ | [Official](https://github.com/pytorch/fairseq)               |                |
| [CMLM+LAT+4 iterations](https://arxiv.org/pdf/2011.06132v1.pdf) | $33.26$ | [Official](https://github.com/shawnkx/NAT-with-Local-AT)     |                |
| [FlowSeq-large](https://arxiv.org/pdf/1909.02480v3.pdf)      | $32.91$ | [Official](https://github.com/XuezheMax/flowseq)             |                |

## Citation

```
  @InProceedings{bojar-EtAl:2016:WMT1,
  author    = {Bojar, Ond{r}ej  and  Chatterjee, Rajen  and  Federmann, Christian  and  Graham, Yvette  and  Haddow, Barry  and  Huck, Matthias  and  Jimeno Yepes, Antonio  and  Koehn, Philipp  and  Logacheva, Varvara  and  Monz, Christof  and  Negri, Matteo  and  Neveol, Aurelie  and  Neves, Mariana  and  Popel, Martin  and  Post, Matt  and  Rubino, Raphael  and  Scarton, Carolina  and  Specia, Lucia  and  Turchi, Marco  and  Verspoor, Karin  and  Zampieri, Marcos},
  title     = {Findings of the 2016 Conference on Machine Translation},
  booktitle = {Proceedings of the First Conference on Machine Translation},
  month     = {August},
  year      = {2016},
  address   = {Berlin, Germany},
  publisher = {Association for Computational Linguistics},
  pages     = {131--198},
  url       = {http://www.aclweb.org/anthology/W/W16/W16-2301}
}
```