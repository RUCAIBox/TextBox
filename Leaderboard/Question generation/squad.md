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
'Denver Broncos [SEP] Super Bowl 50 was an American football game to determine the champion of the National Football League ( NFL ) for the 2015 season . The American Football Conference ( AFC ) champion Denver Broncos defeated the National Football Conference ( NFC ) champion Carolina Panthers 24 – 10 to earn their third Super Bowl title . The game was played on February 7 , 2016 , at Levi \' s Stadium in the San Francisco Bay Area at Santa Clara , California . As this was the 50th Super Bowl , the league emphasized the " golden anniversary " with various gold - themed initiatives , as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals ( under which the game would have been known as " Super Bowl L " ) , so that the logo could prominently feature the Arabic numerals 50 .'
```

Output

```
['Which NFL team represented the AFC at Super Bowl 50 ?']
```

## LeaderBoard

Descending order by BLEU-4.

| Model                                                     | BLEU-4  | Repository                                          | Generated Text |
| --------------------------------------------------------- | ------- | --------------------------------------------------- | -------------- |
| [ ERNIE-GENLARGE](https://arxiv.org/pdf/2001.11314v3.pdf) | $25.41$ | [Official](https://github.com/PaddlePaddle/ERNIE)   |                |
| [UniLMv2](https://arxiv.org/pdf/2002.12804v1.pdf)         | $24.43$ | [Official](https://github.com/microsoft/unilm)      |                |
| [ProphetNet](https://arxiv.org/pdf/2001.04063v3.pdf)      | $23.91$ | [Official](https://github.com/microsoft/ProphetNet) |                |

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

 