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

Input

```
'By HOLLY RAMER, Associated Press\n\nCONCORD, N.H. -- A sick American engineer who was successfully evacuated from the South Pole to New Zealand is awaiting the results of medical tests after having what doctors believed was a stroke in August.\n\nRenee-Nicole Douceur told The Associated Press in an email Tuesday that she had MRI and echocardiogram exams. She said results will be shared with doctors in the United States, "so everyone will be on the same page."\n\n"Back at hotel now to chill out," Douceur wrote. She added, "So nice to see green and smell freshly cut grass, flowers, birds chirping, insects, etc., since it\'s now been over a year on the flat polar plateau of just ice and snow." She landed in Christchurch on Monday.\n\nDouceur, 58, is a Seabrook, N.H., resident who worked as a manager for research station contractor Raytheon Polar Services Co. She asked for an emergency evacuation after having what doctors believed was a stroke in August, but officials rejected her request because of bad weather, saying that sending a rescue plane was too dangerous and that her condition wasn\'t life-threatening.\n\nDoctors she contacted for a second opinion say a tumor may have caused her vision and speech problems.\n\nAfter initially having half her field of vision vanish, Douceur said last week she can now read if she concentrates on just a few words at a time. She said she sometimes jumbles words and has had trouble remembering simple lists of words during medical evaluations.'
```

Output

```
'By HOLLY RAMER, Associated Press CONCORD, N.H. -- A sick American engineer who was successfully evacuated from the South Pole to New Zealand is awaiting the results of medical tests after having what doctors believed was a stroke in August.'
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