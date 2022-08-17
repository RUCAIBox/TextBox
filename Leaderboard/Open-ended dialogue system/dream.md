# DREAM

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/Q19-1014.pdf)

Homepage: [Official](https://dataset.org/dream/)

DREAM is a multiple-choice Dialogue-based REAding comprehension exaMination dataset. In contrast to existing reading comprehension datasets, DREAM is the first to focus on in-depth multi-turn multi-party dialogue understanding.

DREAM contains 10,197 multiple choice questions for 6,444 dialogues, collected from English-as-a-foreign-language examinations designed by human experts. DREAM is likely to present significant challenges for existing reading comprehension systems: 84% of answers are non-extractive, 85% of questions require reasoning beyond a single sentence, and 34% of questions also involve commonsense knowledge.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| DREAM   | $14,264$  | $4,709$   | $4,766$  | $75.6$              | $13.6$              |

### Data Sample

dialogue_id

```
5-510
```

dialogue

```
[ "M: I am considering dropping my dancing class. I am not making any progress.", "W: If I were you, I stick with it. It's definitely worth time and effort." ]
```

 question

```
What does the man suggest the woman do?
```

choice

```
[ "Consult her dancing teacher.", "Take a more interesting class.", "Continue her dancing class." ]
```

answer

```
Continue her dancing class.
```

## LeaderBoard

Descending order by Accuracy.

| Model                                                        | Accuracy | Repository | Generated Text |
| ------------------------------------------------------------ | -------- | ---------- | -------------- |
| [ALBERT-xxlarge + HRCA+ + Multi-Task Learning](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.651.pdf) | $92.6$   |            |                |
| [ALBERT-xxlarge + HRCA+](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.651.pdf) | $91.6$   |            |                |
| [ALBERT-xxlarge + DUMA](https://arxiv.org/abs/2001.09415)    | $90.4$   |            |                |

## Citation

```
 @article{sun-etal-2019-dream,
    title = "{DREAM}: A Challenge Data Set and Models for Dialogue-Based Reading Comprehension",
    author = "Sun, Kai  and
      Yu, Dian  and
      Chen, Jianshu  and
      Yu, Dong  and
      Choi, Yejin  and
      Cardie, Claire",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "7",
    year = "2019",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q19-1014",
    doi = "10.1162/tacl_a_00264",
    pages = "217--231",
}
```