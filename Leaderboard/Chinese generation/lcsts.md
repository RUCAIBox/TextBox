# LCSTS

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1506.05865)

Repository: [Official](http://icrc.hitsz.edu.cn/Article/show/139.html)

It is a large corpus of Chinese short text summarization dataset constructed from the Chinese microblogging website Sina Weibo. This corpus consists of over 2 million real Chinese short texts with short summaries given by the author of each text. Authors also manually tagged the relevance of 10,666 short summaries with their corresponding short texts.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| LCSTS   | $2160531$ | $240060$  | $725$    | -                   | -                   |

### Data Sample

Input

>本文总结了十个可穿戴产品的设计原则，而这些原则，同样也是笔者认为是这个行业最吸引人的地方：1.为人们解决重复性问题；2.从人开始，而不是从机器开始；3.要引起注意，但不要刻意；4.提升用户能力，而不是取代人

Output

>可穿戴技术十大设计原则

## LeaderBoard

Descending order by ROUGE-L.

| Model                                            | Metric | Repository                                 | Generated Text |
| ------------------------------------------------ | ------ | ------------------------------------------ | -------------- |
| [CPT](https://arxiv.org/pdf/2109.05729.pdf)      | $42.0$ | [Official](https://github.com/fastnlp/CPT) |                |
| [ERNIE2.0](https://arxiv.org/pdf/2109.05729.pdf) | $41.4$ |                                            |                |
| [RoBERTa](https://arxiv.org/pdf/2109.05729.pdf)  | $41.0$ |                                            |                |
| [BART](https://arxiv.org/pdf/2109.05729.pdf)     | $40.6$ |                                            |                |
| [mBART](https://arxiv.org/pdf/2109.05729.pdf)    | $37.8$ |                                            |                |
| [mT5](https://arxiv.org/pdf/2109.05729.pdf)      | $36.5$ |                                            |                |
| [CPM-2](https://arxiv.org/pdf/2109.05729.pdf)    | $35.9$ |                                            |                |

## Citation

```
@inproceedings{lcsts,
    title = "{LCSTS}: A Large Scale {C}hinese Short Text Summarization Dataset",
    author = "Hu, Baotian  and
      Chen, Qingcai  and
      Zhu, Fangze",
    booktitle = "Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D15-1229",
    doi = "10.18653/v1/D15-1229",
    pages = "1967--1972",
}
```