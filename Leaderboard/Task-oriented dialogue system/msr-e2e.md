# MSR-E2E

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/pdf/1807.11125v2.pdf)

Repository: [Official](https://github.com/xiul-msr/e2e_dialog_challenge)

This dataset is human-annotated conversational data in three domains (movie-ticket booking, restaurant reservation, and taxi booking).

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| MSR-E2E | $103,362$ | $5,235$   | -        | $51.3$              | $12.8$              |

### Data Sample

Input

```
'Belief state [X_SEP] I would like to have a taxi pick up tomorrow morning at 7 am at 1231 Church St in Reading PA'
```

Output

```
'[taxi] date tomorrow morning pickup time 7am pickup location 1231 Church St pickup location city Reading state PA'
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
 @article{li2018microsoft,
  title={Microsoft Dialogue Challenge: Building End-to-End Task-Completion Dialogue Systems},
  author={Li, Xiujun and Panda, Sarah and Liu, Jingjing and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1807.11125},
  year={2018}
}

@article{li2016user,
  title={A User Simulator for Task-Completion Dialogues},
  author={Li, Xiujun and Lipton, Zachary C and Dhingra, Bhuwan and Li, Lihong and Gao, Jianfeng and Chen, Yun-Nung},
  journal={arXiv preprint arXiv:1612.05688},
  year={2016}
}
```