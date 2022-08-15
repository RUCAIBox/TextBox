# XSum

## Dataset

### Instruction

Paper: [Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://arxiv.org/abs/1808.08745)

Repository: [Official](https://github.com/EdinburghNLP/XSum)

The Extreme Summarization (XSum) dataset is a dataset for evaluation of abstractive single-document summarization systems. The goal is to create a short, one-sentence new summary answering the question “What is the article about?”. The articles are collected from BBC articles (2010 to 2017) and cover a wide variety of domains (e.g., News, Politics, Sports, Weather, Business, Technology, Science, Health, Family, Education, Entertainment and Arts).

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| XSum    | $204,045$ | $11,332$  | $11,334$ | $373.7$             | $21.1$              |

### Data Sample

Input
```
The problem is affecting people using the older versions of the PlayStation 3, called the "Fat" model.The problem isn't affecting the newer PS3 Slim systems that have been on sale since September last year.Sony have also said they are aiming to have the problem fixed shortly but is advising some users to avoid using their console for the time being."We hope to resolve this problem within the next 24 hours," a statement reads. "In the meantime, if you have a model other than the new slim PS3, we advise that you do not use your PS3 system, as doing so may result in errors in some functionality, such as recording obtained trophies, and not being able to restore certain data."We believe we have identified that this problem is being caused by a bug in the clock functionality incorporated in the system."The PlayStation Network is used by millions of people around the world.It allows users to play their friends at games like Fifa over the internet and also do things like download software or visit online stores.
```
Output
```
Sony has told owners of older models of its PlayStation 3 console to stop using the machine because of a problem with the PlayStation Network.
```
## LeaderBoard

Descending order by ROUGE-2.

| Model                                                     | ROUGE-1 | ROUGE-2 | ROUGE-L | Repository                                                   | Generated Text                                               |
| --------------------------------------------------------- | ------- | ------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [BRIO](https://arxiv.org/abs/2203.16804)                  | $49.07$ | $25.59$ | $40.40$ | [Official](https://github.com/yixinL7/BRIO)                  |                                                              |
| [PEGASUS+SummaReranker](https://arxiv.org/abs/2203.06569) | $48.12$ | $24.95$ | $40.00$ | [Official](https://github.com/ntunlp/SummaReranker)          |                                                              |
| [SimCLS](https://aclanthology.org/2021.acl-short.135.pdf) | $47.61$ | $24.57$ | $29.44$ | [Official](https://github.com/yixinL7/SimCLS)                |                                                              |
| [PEGASUS](https://arxiv.org/abs/1912.08777)               | $47.21$ | $24.56$ | $39.25$ | [Official](https://github.com/google-research/pegasus)       | [Corpus](https://drive.google.com/drive/folders/1tpsI1aBMHaj8h_eOLWqvXDaCkmqmqEaF?usp=sharing) |
| [SeqCo](https://arxiv.org/abs/2109.03481)                 | $45.65$ | $22.41$ | $37.04$ | [Official](https://github.com/xssstory/SeqCo)                |                                                              |
| [BART](https://arxiv.org/abs/1910.13461)                  | $45.14$ | $22.27$ | $37.25$ | [Official](https://github.com/facebookresearch/fairseq/tree/main/examples/bart) | [Corpus](https://drive.google.com/drive/folders/1k1eROTpSe9cvoKLT1v3wXd_BRKME9ROn?usp=sharing) |
| [GSum](https://arxiv.org/abs/2010.08014)                  | $45.40$ | $21.89$ | $36.67$ | [Official](https://github.com/neulab/guided_summarization)   | [Corpus](https://drive.google.com/drive/folders/1lfGRNkP0dxb9oRlIi0hO4zvYG_hUtcX8?usp=sharing) |
| [BertSumExtAbs](https://arxiv.org/abs/1908.08345)         | $38.81$ | $16.50$ | $31.27$ | [Official](https://github.com/nlpyang/PreSumm)               |                                                              |
| [T-CONVS2S](https://aclanthology.org/D18-1206.pdf)        | $31.89$ | $11.54$ | $25.75$ | [Official](https://github.com/EdinburghNLP/XSum)             | [Corpus](https://drive.google.com/drive/folders/1svxHYvtV_wCOZpsoV7V-NHFTme3LHzJa?usp=sharing) |
| [PTGEN](https://aclanthology.org/D18-1206.pdf)            | $29.70$ | $9.21$  | $23.24$ | [Official](https://github.com/EdinburghNLP/XSum)             |                                                              |

## Citation
```
@inproceedings{xsum,
    title = "Don{'}t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization",
    author = "Narayan, Shashi  and
      Cohen, Shay B.  and
      Lapata, Mirella",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1206",
    doi = "10.18653/v1/D18-1206",
    pages = "1797--1807",
}
```