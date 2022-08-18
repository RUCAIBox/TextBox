# WMT16 German-English

## Dataset

### Instruction

Paper: [Paper](http://www.aclweb.org/anthology/W/W16/W16-2301)

Homepage: [Official](https://www.statmt.org/wmt16/index.html)

Translate dataset based on the data from statmt.org.

### Overview

| Dataset              | Num Train   | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------------------- | ----------- | --------- | -------- | ------------------- | ------------------- |
| WMT16 German-English | $4,548,885$ | $2,169$   | $2,999$  |                     |                     |

### Data Sample

English

```
Resumption of the session
```

German

```
Wiederaufnahme der Sitzungsperiode
```

## LeaderBoard

### English $\rarr$ German

Descending order by BLEU.

| Model                                                        | BLEU    | Repository                                             | Generated Text |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------ | -------------- |
| [MADL](https://openreview.net/pdf?id=HyGhN2A5tm)             | $40.68$ |                                                        |                |
| [Attentional encoder-decoder + BPE](https://arxiv.org/pdf/1606.02891v2.pdf) | $34.2$  | [Official](https://github.com/rsennrich/wmt16-scripts) |                |
| [Linguistic Input Features](https://arxiv.org/pdf/1606.02892v2.pdf) | $28.4$  | [Official](https://github.com/rsennrich/wmt16-scripts) |                |

#### German $\rarr$ English

Descending order by BLEU.

| Model                                                        | BLEU   | Repository                                             | Generated Text |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------ | -------------- |
| [FLAN 137B zero-shot](https://arxiv.org/pdf/2109.01652v5.pdf) | $39.8$ | [Official](https://github.com/google-research/flan)    |                |
| [Attentional encoder-decoder + BPE](https://arxiv.org/pdf/1606.02891v2.pdf) | $38.6$ | [Official](https://github.com/rsennrich/wmt16-scripts) |                |
| [Linguistic Input Features](https://arxiv.org/pdf/1606.02892v2.pdf) | $32.9$ | [Official](https://github.com/rsennrich/wmt16-scripts) |                |

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