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
"i like to garden. [SEP] Gardening is the practice of growing and cultivating plants as part of horticulture. [SEP] In gardens, ornamental plants are often grown for their flowers, foliage, or overall appearance; useful plants, such as root vegetables, leaf vegetables, fruits, and herbs, are grown for consumption, for use as dyes, or for medicinal or cosmetic use. [SEP] Gardening is considered by many people to be a relaxing activity. [SEP] Gardening ranges in scale from fruit orchards, to long boulevard plantings with one or more different types of shrubs, trees, and herbaceous plants, to residential yards including lawns and foundation plantings, to plants in large or small containers grown inside or outside. [SEP] Gardening may be very specialized, with only one type of plant grown, or involve a large number of different plants in mixed plantings. [SEP] It involves an active participation in the growing of plants, and tends to be labor-intensive, which differentiates it from farming or forestry. [SEP] Forest gardening, a forest-based food production system, is the world's oldest form of gardening. [SEP] Forest gardens originated in prehistoric times along jungle-clad river banks and in the wet foothills of monsoon regions. [SEP] In the gradual process of families improving their immediate environment, useful tree and vine species were identified, protected and improved while undesirable species were eliminated. [X_SEP] I like Gardening, even when I've only been doing it for a short time."
```

Output

```
'I live on a farm, we garden all year long, it is very relaxing.'
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