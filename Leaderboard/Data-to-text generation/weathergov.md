# WEATHERGOV

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/P09-1011.pdf)

This dataset is about weather forecasts. In this dataset, there are two part for each city and date, one for the day forecast and one for the night forecast. The forecasts consist of hour-by-hour measurements of temperature, wind speed, sky cover, chance of rain, etc., which represent the underlying world state.

### Overview

| Dataset    | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ---------- | --------- | --------- | -------- | ------------------- | ------------------- |
| WEATHERGOV | $25,000$  | $1,000$   | $3,528$  | $148.7$             | $30.6$              |

### Data Sample

Input

```
'type | skyCover [SEP] mode | 0-25 [SEP] date | 2009-02-08 [SEP] label | Tonight [SEP] time | 17-30 [X_SEP] type | temperature [SEP] min | 33 [SEP] mean | 44 [SEP] max | 60 [SEP] date | 2009-02-08 [SEP] label | Tonight [SEP] time | 17-30 [X_SEP] type | windDir [SEP] mode | NNW [SEP] date | 2009-02-08 [SEP] label | Tonight [SEP] time | 17-30 [SEP] type | windSpeed [SEP] min | 2 [SEP] mean | 5 [SEP] max | 13 [SEP] mode | 0-10 [SEP] date | 2009-02-08 [SEP] label | Tonight [SEP] time | 17-30'
```

Output

```
'Clear , with a low around 31 . North wind between 6 and 9 mph becoming calm .'
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
 @inproceedings{liang-etal-2009-learning,
    title = "Learning Semantic Correspondences with Less Supervision",
    author = "Liang, Percy  and
      Jordan, Michael  and
      Klein, Dan",
    booktitle = "Proceedings of the Joint Conference of the 47th Annual Meeting of the {ACL} and the 4th International Joint Conference on Natural Language Processing of the {AFNLP}",
    month = aug,
    year = "2009",
    address = "Suntec, Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P09-1011",
    pages = "91--99",
}
```