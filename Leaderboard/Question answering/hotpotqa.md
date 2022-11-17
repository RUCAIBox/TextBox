# HotpotQA

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1809.09600)

Homepage: [Official](https://hotpotqa.github.io/)

Repository: [Official](https://github.com/hotpotqa/hotpot)

HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

### Overview

| Dataset  | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | -------- | ------------------- | ------------------- |
| HotpotQA | $90,447$  | $7,405$   | -        | $187.9$             | $2.2$               |

### Data Sample

Input

```
'Were Scott Derrickson and Ed Wood of the same nationality? [X_SEP] Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. He is best known for directing horror films such as "Sinister", "The Exorcism of Emily Rose", and "Deliver Us From Evil", as well as the 2016 Marvel Cinematic Universe installment, "Doctor Strange." [SEP] Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.'
```

Output

```
'yes'
```

## LeaderBoard

Descending order by Answer-F1.

| Model                                                        | Answer-EM | Answer-F1 | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | --------- | --------- | ------------------------------------------------------------ | -------------- |
| [BigBird-etc](https://arxiv.org/pdf/2007.14062v2.pdf)        | -         | $81.2$    | [Official](https://github.com/google-research/bigbird)       |                |
| [ AISO](https://arxiv.org/pdf/2109.06747v1.pdf)              | $67.5$    | $80.5$    | [Official](https://github.com/zycdev/aiso)                   |                |
| [ HopRetriever + Sp-search](https://arxiv.org/pdf/2012.15534v1.pdf) | $67.1$    | $79.9$    |                                                              |                |
| [TPRR](http://playbigdata.ruc.edu.cn/dou/publication/2021_SIGIR_Ranker.pdf) | $67.0$    | $79.5$    |                                                              |                |
| [IRRR+](https://arxiv.org/pdf/2010.12527v4.pdf)              | $66.3$    | $79.1$    | [Official](https://github.com/beerqa/irrr)                   |                |
| [DDRQA](https://arxiv.org/pdf/2009.07465v5.pdf)              | $62.5$    | $75.9$    |                                                              |                |
| [Recursive Dense Retriever](https://arxiv.org/pdf/2009.12756v2.pdf) | $62.3$    | $75.3$    | [Official](https://github.com/facebookresearch/multihop_dense_retrieval) |                |

## Citation

```
@inproceedings{hotpotqa,
    title = "{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering",
    author = "Yang, Zhilin  and
      Qi, Peng  and
      Zhang, Saizheng  and
      Bengio, Yoshua  and
      Cohen, William  and
      Salakhutdinov, Ruslan  and
      Manning, Christopher D.",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1259",
    doi = "10.18653/v1/D18-1259",
    pages = "2369--2380",
}
```
