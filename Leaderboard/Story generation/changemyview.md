# ChangeMyView

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/2020.emnlp-main.57.pdf)

This dataset contains pairs of original post (OP) statement on a controversial issue about politics and filtered high-quality counter-arguments, covering 14, 833 threads from 2013 to 2018.

### Overview

| Dataset      | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------------ | --------- | --------- | -------- | ------------------- | ------------------- |
| ChangeMyView | $42,462$  | $6,480$   | $7,562$  | $17.9$              | $104.1$             |

### Data Sample

Input

```
'CMV : Military desertion should not be a crime as it is an entirely normal human behavior and invokes on the right of freedom of speech and prosecution of such can often be observed as external coercion which goes against freedom of action.'
```

Output

```
"In order for your argument to be true, you'd also have to say that dodging a draft wouldn't be a crime either, right? It'd be a part of your right to free speech. And if you could dodge the draft during a time of extreme crisis, how would you ensure that the country could be safely protected?"
```

## LeaderBoard

Descending order by METRIC.

| Model | Metric | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
|       |        |            |                |
|       |        |            |                |
|       |        |            |                |

## Citation

```
@inproceedings{hua-wang-2020-pair,
    title = "{PAIR}: Planning and Iterative Refinement in Pre-trained Transformers for Long Text Generation",
    author = "Hua, Xinyu  and
      Wang, Lu",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.57",
    doi = "10.18653/v1/2020.emnlp-main.57",
    pages = "781--793",
}
```

