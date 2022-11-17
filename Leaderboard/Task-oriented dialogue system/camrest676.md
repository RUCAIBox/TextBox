# CamRest676

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/E17-1042.pdf)

Repository: [Official]()

CamRest676 Human2Human dataset contains the following three json files:1. CamRest676.json: the woz dialogue dataset, which contains the conversion from users and wizards, as well as a set of coarse labels for each user turn.2. CamRestDB.json: the Cambridge restaurant database file, containing restaurants in the Cambridge UK area and a set of attributes.3. The ontology file, specific all the values the three informable slots can take.

### Overview

| Dataset    | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ---------- | --------- | --------- | -------- | ------------------- | ------------------- |
| CamRest676 | $4,872$   | $616$     | -        | $55.3$              | $9.4$               |

### Data Sample

Input

```
'Belief state [X_SEP] I want to find a restaurant in the south part of town serving singaporean food.'
```

Output

```
'[restaurant] food singaporean area south'
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
@inproceedings{wen-etal-2017-network,
    title = "A Network-based End-to-End Trainable Task-oriented Dialogue System",
    author = "Wen, Tsung-Hsien  and
      Vandyke, David  and
      Mrk{\v{s}}i{\'c}, Nikola  and
      Ga{\v{s}}i{\'c}, Milica  and
      Rojas-Barahona, Lina M.  and
      Su, Pei-Hao  and
      Ultes, Stefan  and
      Young, Steve",
    booktitle = "Proceedings of the 15th Conference of the {E}uropean Chapter of the Association for Computational Linguistics: Volume 1, Long Papers",
    month = apr,
    year = "2017",
    address = "Valencia, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/E17-1042",
    pages = "438--449",
}
```

