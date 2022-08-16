# SQuAD

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/D16-1264.pdf)

Homepage: [Official](https://rajpurkar.github.io/SQuAD-explorer/)

**S**tanford **Qu**estion **A**nswering **D**ataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| SQuAD   | $75,722$  | $10,570$  | $11,877$ | $156.2$             | $3.6$               |

### Data Sample

```
{
    "answers": {
        "answer_start": [1],
        "text": ["This is a test text"]
    },
    "context": "This is a test context.",
    "id": "1",
    "question": "Is this a test?",
    "title": "train test"
}
```

## LeaderBoard

Descending order by EM.

| Model                                              | EM       | F1       | Repository                                               | Generated Text |
| -------------------------------------------------- | -------- | -------- | -------------------------------------------------------- | -------------- |
| [ LUKE](https://arxiv.org/pdf/2010.01057v1.pdf)    | $90.202$ | $95.379$ | [Official](https://github.com/studio-ousia/luke)         |                |
| [ XLNet](https://arxiv.org/pdf/1906.08237v2.pdf)   | $89.898$ | $95.080$ | [Official](https://github.com/zihangdai/xlnet)           |                |
| [SpanBERT](https://arxiv.org/pdf/1907.10529v3.pdf) | $88.8$   | $94.6$   | [Official](https://github.com/facebookresearch/SpanBERT) |                |

## Citation

```
@inproceedings{rajpurkar-etal-2016-squad,
    title = "{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text",
    author = "Rajpurkar, Pranav  and
      Zhang, Jian  and
      Lopyrev, Konstantin  and
      Liang, Percy",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D16-1264",
    doi = "10.18653/v1/D16-1264",
    pages = "2383--2392",
}
```

