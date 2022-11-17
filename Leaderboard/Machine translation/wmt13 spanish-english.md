# WMT13 Spanish-English

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/W13-2201.pdf)

Homepage: [Official](https://www.statmt.org/wmt13/)

Translate dataset based on the data from statmt.org.

### Overview

| Dataset               | Num Train    | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| --------------------- | ------------ | --------- | -------- | ------------------- | ------------------- |
| WMT13 Spanish-English | $11,196,913$ | $13,573$  | -        | 29.3                | 25.1                |

### Data Sample

English

```
In that connection the High Commissioner recalled that the General Assembly, in its resolution 52/111, decided to convene a World Conference against Racism, Racial Discrimination, Xenophobia and Related Intolerance, the preparation of which is being coordinated by the Office of the United Nations High Commissioner for Human Rights.
```

Spanish

```
A ese respecto, recordó que la Asamblea General, en su resolución 52/111, había decidido convocar una conferencia mundial contra el racismo, la discriminación racial, la xenofobia y las formas conexas de intolerancia, de cuya preparación se estaba encargando la Oficina del Alto Comisionado de las Naciones Unidas para los Derechos Humanos.
```

## LeaderBoard

#### Spanish $\rarr$ English

Descending order by BLEU.

| Model | BLEU | Repository | Generated Text |
| ----- | ---- | ---------- | -------------- |
|       |      |            |                |
|       |      |            |                |
|       |      |            |                |

#### English $\rarr$ Spanish

Descending order by BLEU.

| Model                                         | BLEU   | Repository                                                   | Generated Text |
| --------------------------------------------- | ------ | ------------------------------------------------------------ | -------------- |
| [mBART](https://arxiv.org/pdf/2001.08210.pdf) | $34.0$ | [Official](https://github.com/pytorch/fairseq/tree/master/examples/mbart) |                |

## Citation

```
 @inproceedings{bojar-etal-2013-findings,
    title = "Findings of the 2013 {W}orkshop on {S}tatistical {M}achine {T}ranslation",
    author = "Bojar, Ond{\v{r}}ej  and
      Buck, Christian  and
      Callison-Burch, Chris  and
      Federmann, Christian  and
      Haddow, Barry  and
      Koehn, Philipp  and
      Monz, Christof  and
      Post, Matt  and
      Soricut, Radu  and
      Specia, Lucia",
    booktitle = "Proceedings of the Eighth Workshop on Statistical Machine Translation",
    month = aug,
    year = "2013",
    address = "Sofia, Bulgaria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W13-2201",
    pages = "1--44",
}
```