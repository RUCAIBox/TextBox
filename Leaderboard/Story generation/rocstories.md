# ROCStories

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/N16-1098.pdf)

Homepage: [Official](https://cs.rochester.edu/nlp/rocstories/)

ROCStories is a collection of commonsense short stories. The corpus consists of 100,000 five-sentence stories. Each story logically follows everyday topics created by Amazon Mechanical Turk workers. These stories contain a variety of commonsense causal and temporal relations between everyday events. Writers also develop an additional 3,742 Story Cloze Test stories which contain a four-sentence-long body and two candidate endings. The endings were collected by asking Mechanical Turk workers to write both a right ending and a wrong ending after eliminating original endings of given short stories. Both endings were required to make logical sense and include at least one character from the main story line. 

### Overview

| Dataset    | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ---------- | --------- | --------- | -------- | ------------------- | ------------------- |
| ROCStories | $176,688$ | $9,816$   | $4,909$  | $9.0$               | $40.7$              |

### Data Sample

Title

```
The Test
```

Five-sentence Story

```
Jennifer has a big exam tomorrow. She got so stressed, she pulled an all-nighter. She went into class the next day, weary as can be. Her teacher stated that the test is postponed for next week. Jennifer felt bittersweet about it.
```

## LeaderBoard

Descending order by BLEU-1.

| Model                                                        | BLEU-1 | Perplexity | Repository                                                 | Generated Text |
| ------------------------------------------------------------ | ------ | ---------- | ---------------------------------------------------------- | -------------- |
| [Beam search + A*esque](https://arxiv.org/pdf/2112.08726v1.pdf) | $34.4$ | $2.14$     | [Official](https://github.com/GXimingLu/a_star_neurologic) |                |

## Citation

```
@inproceedings{mostafazadeh-etal-2016-corpus,
    title = "A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories",
    author = "Mostafazadeh, Nasrin  and
      Chambers, Nathanael  and
      He, Xiaodong  and
      Parikh, Devi  and
      Batra, Dhruv  and
      Vanderwende, Lucy  and
      Kohli, Pushmeet  and
      Allen, James",
    booktitle = "Proceedings of the 2016 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N16-1098",
    doi = "10.18653/v1/N16-1098",
    pages = "839--849",
}
```

