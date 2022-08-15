# LogicNLG

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/2004.10404)

Repository: [Official](https://github.com/wenhuchen/LogicNLG)

It's a dataset based on TabFact (Chen et al., 2019), which is a table-based fact-checking dataset with rich logical inferences in the annotated statements.

### Overview

| Dataset  | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | -------- | ------------------- | ------------------- |
| LogicNLG | $28,450$  | $4,260$   | $4,305$  | $178.4$             | $14.2$              |

### Data Sample

Input

```

```

Output

```

```

## LeaderBoard

Descending order by BLEU-1.

| Model                                          | BLEU-1 | BLEU-2 | BLEU-3 | Repository | Generated Text |
| ---------------------------------------------- | ------ | ------ | ------ | ---------- | -------------- |
| [DCVED](https://arxiv.org/abs/2004.10404)      | $49.5$ | $28.6$ | $15.3$ |            |                |
| [GPT-TabGen](https://arxiv.org/abs/2004.10404) | $48.8$ | $27.1$ | $12.6$ |            |                |


## Citation

```
 @inproceedings{logicnlg,
    title = "Logical Natural Language Generation from Open-Domain Tables",
    author = "Chen, Wenhu  and
      Chen, Jianshu  and
      Su, Yu  and
      Chen, Zhiyu  and
      Wang, William Yang",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.708",
    doi = "10.18653/v1/2020.acl-main.708",
    pages = "7929--7942",
}
```