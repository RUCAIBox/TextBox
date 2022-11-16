HotpotQA

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
'yes [X_SEP] Scott Derrickson ( born July 16 , 1966 ) is an American director , screenwriter and producer . Edward Davis Wood Jr. ( October 10 , 1924 – December 10 , 1978 ) was an American filmmaker , actor , writer , producer , and director .'
```

Output

```
'Were Scott Derrickson and Ed Wood of the same nationality ?'
```

## LeaderBoard

Descending order by METRIC.

| Model | METRIC | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
|       |        |            |                |
|       |        |            |                |
|       |        |            |                |

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