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

```
[
	{
		"_id":"5a8b57f25542995d1e6f1371",
		"answer":"yes",
		"question":"Were Scott Derrickson and Ed Wood of the same nationality?",
		"supporting_facts":[["Scott Derrickson",0],["Ed Wood",0]],
		"context":[["Ed Wood (film)",["Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood."," The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau."," Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast."]], ... ]
	}
	...
]
```

## LeaderBoard

Descending order by F1.

| Model                                                        | Answer-EM | Answer-F1 | Repository                                 | Generated Text |
| ------------------------------------------------------------ | --------- | --------- | ------------------------------------------ | -------------- |
| [ AISO](https://arxiv.org/pdf/2109.06747v1.pdf)              | $67.5$    | $80.5$    | [Official](https://github.com/zycdev/aiso) |                |
| [ HopRetriever + Sp-search](https://arxiv.org/pdf/2012.15534v1.pdf) | $67.1$    | $79.9$    |                                            |                |
| [IRRR+](https://arxiv.org/pdf/2010.12527v4.pdf)              | $66.3$    | $79.1$    | [Official](https://github.com/beerqa/irrr) |                |

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
