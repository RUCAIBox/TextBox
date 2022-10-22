# WikiBio

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/pdf/1603.07771v3.pdf)

Homepage: [Official](https://rlebret.github.io/wikipedia-biography-dataset/)

This dataset gathers 728,321 biographies from English Wikipedia. It aims at evaluating text generation algorithms. For each article, we provide the first paragraph and the infobox (both tokenized).

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| WikiBio | $582,659$ | $72,831$  | $72,831$ | $81.6$              | $26.1$              |

### Data Sample

Input

```
'name | lenny randle [SEP] position | second baseman / third baseman [SEP] bats | switch [SEP] throws | right [SEP] birth_date | 12 february 1949 [SEP] birth_place | long beach , california [SEP] debutdate | june 16 [SEP] debutyear | 1971 [SEP] debutteam | washington senators [SEP] finaldate | june 20 [SEP] finalyear | 1982 [SEP] finalteam | seattle mariners [SEP] statlabel | batting average home runs runs batted in [SEP] statvalue | .257 27 322 [SEP] article_title | lenny randle'
```

Output

```
'leonard shenoff randle ( born february 12 , 1949 ) is a former major league baseball player .'
```

## LeaderBoard

Descending order by BLEU.

| Model                                                        | BLEU    | Repository                                              | Generated Text |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------- | -------------- |
| [Field-gating Seq2seq + dual attention + beam search](https://arxiv.org/pdf/1711.09724v1.pdf) | $44.71$ | [Official](https://github.com/tyliupku/wiki2bio)        |                |
| [ MBD](https://arxiv.org/pdf/2102.02810v2.pdf)               | $41.56$ | [Official](https://github.com/KaijuML/dtt-multi-branch) |                |
| [Table NLM](https://arxiv.org/pdf/1603.07771v3.pdf)          | $34.70$ |                                                         |                |
| [Template Kneser-Ney](https://arxiv.org/pdf/1603.07771v3.pdf) | $19.8$  |                                                         |                |

## Citation

```
 @inproceedings{wikibio,
    title = "Neural Text Generation from Structured Data with Application to the Biography Domain",
    author = "Lebret, R{\'e}mi  and
      Grangier, David  and
      Auli, Michael",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D16-1128",
    doi = "10.18653/v1/D16-1128",
    pages = "1203--1213",
}
```