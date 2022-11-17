# SQuAD

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/D16-1264.pdf)

Homepage: [Official](https://rajpurkar.github.io/SQuAD-explorer/)

**S**tanford **Qu**estion **A**nswering **D**ataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| SQuAD   | $75,722$  | $10,570$  | $11,877$ | $156.2$             | $3.6$               |

### Data Sample

Input

```
'Which NFL team represented the AFC at Super Bowl 50? [X_SEP] Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.'
```

Output

```
'Denver Broncos'
```

## LeaderBoard

Descending order by F1.

| Model                                                        | EM       | F1       | Repository                                                  | Generated Text |
| ------------------------------------------------------------ | -------- | -------- | ----------------------------------------------------------- | -------------- |
| [ LUKE](https://arxiv.org/pdf/2010.01057v1.pdf)              | $90.202$ | $95.379$ | [Official](https://github.com/studio-ousia/luke)            |                |
| [ XLNet](https://arxiv.org/pdf/1906.08237v2.pdf)             | $89.898$ | $95.080$ | [Official](https://github.com/zihangdai/xlnet)              |                |
| [SpanBERT](https://arxiv.org/pdf/1907.10529v3.pdf)           | $88.8$   | $94.6$   | [Official](https://github.com/facebookresearch/SpanBERT)    |                |
| [BERT](https://arxiv.org/pdf/1810.04805v2.pdf)               | $87.433$ | $93.160$ |                                                             |                |
| [LinkBERT](https://arxiv.org/pdf/2203.15827v1.pdf)           | $87.45$  | $92.7$   | [Official](https://github.com/michiyasunaga/LinkBERT)       |                |
| [MAMCN+](https://aclanthology.org/W18-2603.pdf)              | $79.692$ | $86.727$ |                                                             |                |
| [Reinforced Mnemonic Reader](https://arxiv.org/pdf/1705.02798v6.pdf) | $79.545$ | $86.654$ |                                                             |                |
| [BiDAF + Self Attention + ELMo](https://arxiv.org/pdf/1802.05365v2.pdf) | $78.58$  | $85.833$ |                                                             |                |
| [MEMEN](https://arxiv.org/pdf/1707.09098v1.pdf)              | $78.234$ | $85.344$ |                                                             |                |
| [SAN](https://arxiv.org/pdf/1712.03556v2.pdf)                | $76.828$ | $84.396$ |                                                             |                |
| [RaSoR + TR + LM](https://arxiv.org/pdf/1712.03609v4.pdf)    | $77.583$ | $84.163$ |                                                             |                |
| [FusionNet](https://arxiv.org/pdf/1711.07341v2.pdf)          | $75.968$ | $83.900$ | [Official](https://github.com/hsinyuan-huang/FusionNet-NLI) |                |
| [KAR](https://arxiv.org/pdf/1809.03449v3.pdf)                | $76.125$ | $83.538$ |                                                             |                |


## Citation

```
@inproceedings{rajpurkar-etal-2016-squad,
    title = "{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text",
    author = "Rajpurkar, Pranav  and
      Zhang, Jian  and
      Lopyrev, Konstantin  and
      Liang, Percy",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D16-1264",
    doi = "10.18653/v1/D16-1264",
    pages = "2383--2392",
}
```

