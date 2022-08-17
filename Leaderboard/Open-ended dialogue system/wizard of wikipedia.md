# Wizard of Wikipedia

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1811.01241)

Homepage: [Official](https://parl.ai/projects/wizard_of_wikipedia/)

Wizard of Wikipedia is a large dataset with conversations directly grounded with knowledge retrieved from Wikipedia. It is used to train and evaluate dialogue systems for knowledgeable open dialogue with clear grounding.

### Overview

| Dataset             | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------------------- | --------- | --------- | -------- | ------------------- | ------------------- |
| Wizard of Wikipedia | $148,357$ | $15,767$  | $15,564$ | $297.0$             | $16.7$              |

### Data Sample

Input

```

```

Output

```

```

## LeaderBoard

Descending order by Test Seen PPL.

| Model                                                        | Test Seen PPL | Test Unseen PPL | Repository | Generated Text |
| ------------------------------------------------------------ | ------------- | --------------- | ---------- | -------------- |
| [End-to-end Transformer MemNet](https://arxiv.org/abs/1811.01241) | $63.5$        | $97.3$          |            |                |
| [Two-Stage Transformer Memnet](https://arxiv.org/abs/1811.01241) | $46.5$        | $84.8$          |            |                |
| [Vanilla Transformer (no knowledge)](https://arxiv.org/abs/1811.01241) | $41.8$        | $87.0$          |            |                |

## Citation

```
 @inproceedings{wow,
title={Wizard of Wikipedia: Knowledge-Powered Conversational Agents},
author={Emily Dinan and Stephen Roller and Kurt Shuster and Angela Fan and Michael Auli and Jason Weston},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=r1l73iRqKm},
}
```