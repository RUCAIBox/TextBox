# Newsroom

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/N18-1065.pdf)

Repository: [Official](https://github.com/lil-lab/newsroom)

Homepage: [Official](http://lil.nlp.cornell.edu/newsroom/)

CORNELL NEWSROOM is a large dataset for training and evaluating summarization systems. It contains 1.3 million articles and summaries written by authors and editors in the newsrooms of 38 major publications. The summaries are obtained from search and social metadata between 1998 and 2017 and use a variety of summarization strategies combining extraction and abstraction.

### Overview

| Dataset  | Num Train | Num Valid | Num Test  | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | --------- | ------------------- | ------------------- |
| Newsroom | $995,041$ | $108,837$ | $108,862$ | $642.4$             | $26.7$              |

### Data Sample

```
{
    "compression": 64.30769348144531,
    "compression_bin": "high",
    "coverage": 0.8461538553237915,
    "coverage_bin": "medium",
    "date": "20141011140242",
    "density": 9.15384578704834,
    "density_bin": "extractive",
    "summary": "DC's Chief Creative Officer on mobile gaming: \"The reach that these games can get instantly is something short of a miracle.\"",
    "text": "While Marvel Studios DIS has released many more big-budget comic book movies over the past decade, the anticipation is building for Batman vs Superman: Dawn of Justice starring Henry Cavill and Ben Affleck and the long-awaited Justice League movie. DC characters are also all over television this fall with The Flash, Gotham, Constantine and iZombie joining Arrow. And gamers are looking forward to the 2015 release of Batman: Arkham Knight, which marks the next gen debut of Batman. The man at the center of this multimedia DC Universe is Geoff Johns, Chief Creative Officer at DC Entertainment TWX . The writer of Superman, Green Lantern, The Flash, Teen Titans and Justice Society of America has been busy working with other creatives across every medium to connect DC superheroes with a mainstream audience, which is one seemingly more receptive to them than ever before. Before New York Comic Con, Johns recently talked to Fortune about how video games, mobile and digital distribution are [...]",
    "title": "DC Entertainment exec Geoff Johns talks games, tech and TV",
    "url": "http://fortune.com/2014/10/11/geoff-johns-dc-entertainment/"
}
```

## LeaderBoard

Descending order by ROUGE-2.

| Model                                                        | ROUGE-1 | ROUGE-2 | ROUGE-L | Repository | Generated Text |
| ------------------------------------------------------------ | ------- | ------- | ------- | ---------- | -------------- |
| [Modified P-G](https://www.aclweb.org/anthology/N19-4012)    | $39.91$ | $28.38$ | $36.87$ |            |                |
| [C10110/SpaCy](https://arxiv.org/abs/1812.02303)             | $39.36$ | $27.86$ | $36.35$ |            |                |
| [ExtConSumm Extractive](https://arxiv.org/abs/1904.02020)    | $39.40$ | $27.80$ | $36.20$ |            |                |
| [Lede-3 Baseline](https://www.aclweb.org/anthology/N18-1065) | $32.02$ | $21.08$ | $29.59$ |            |                |
| [TLM](https://arxiv.org/abs/1909.03186)                      | $33.24$ | $20.01$ | $29.21$ |            |                |
| [Pointer-Generator](https://arxiv.org/abs/1704.04368)        | $27.54$ | $13.32$ | $23.50$ |            |                |
| [TextRank](https://arxiv.org/abs/1602.03606)                 | $24.45$ | $10.12$ | $20.13$ |            |                |
| [Fast-RL](http://dmkd.cs.vt.edu/papers/SDM19.pdf)            | $21.93$ | $9.37$  | $19.61$ |            |                |
| [Seq2Seq + Attention](https://www.aclweb.org/anthology/D15-1044) | $5.99$  | $0.37$  | $5.41$  |            |                |

## Citation

```
 @inproceedings{grusky-etal-2018-newsroom,
    title = "{N}ewsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies",
    author = "Grusky, Max  and
      Naaman, Mor  and
      Artzi, Yoav",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-1065",
    doi = "10.18653/v1/N18-1065",
    pages = "708--719",
}
```