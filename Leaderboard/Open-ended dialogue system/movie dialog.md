# Movie Dialog

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1511.06931)

Homepage: [Official](https://research.fb.com/downloads/babi/)

Movie Dialog dataset (MDD) is designed to measure how well models can perform at goal and non-goal orientated dialog centered around the topic of movies (question answering, recommendation and discussion).

### Overview

| Dataset      | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------------ | --------- | --------- | -------- | ------------------- | ------------------- |
| Movie Dialog | $762,751$ | $8,216$   | $8,066$  | $126.9$             | $44.0$              |

### Data Sample

Input

```
"Yup. That 's what the villains in X-Men argue . Just like the Japanese were a danger during ww2 so we put them in camps ."
```

Output

```
'Not the same . This is a movie about beings that fly and could decapitated you with their mind . They are not some minority that people are afraid because of their skin colour or some idiot shit . The mutants are really dangerous .'
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
 @inproceedings{md,
  author    = {Jesse Dodge and
               Andreea Gane and
               Xiang Zhang and
               Antoine Bordes and
               Sumit Chopra and
               Alexander H. Miller and
               Arthur Szlam and
               Jason Weston},
  title     = {Evaluating Prerequisite Qualities for Learning End-to-End Dialog Systems},
  booktitle = {4th International Conference on Learning Representations, {ICLR} 2016},
  year      = {2016},
  url={http://arxiv.org/abs/1511.06931},
}
```