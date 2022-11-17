# DREAM

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/Q19-1014.pdf)

Homepage: [Official](https://dataset.org/dream/)

DREAM is a multiple-choice **D**ialogue-based **REA**ding comprehension exa**M**ination dataset. In contrast to existing reading comprehension datasets, DREAM is the first to focus on in-depth multi-turn multi-party dialogue understanding.

DREAM contains 10,197 multiple choice questions for 6,444 dialogues, collected from English-as-a-foreign-language examinations designed by human experts. DREAM is likely to present significant challenges for existing reading comprehension systems: 84% of answers are non-extractive, 85% of questions require reasoning beyond a single sentence, and 34% of questions also involve commonsense knowledge.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| DREAM   | $14,264$  | $4,709$   | $4,766$  | $75.6$              | $13.6$              |

### Data Sample

Input

```
'How long have you been teaching in this middle school?'
```

Output

```
For ten years. To be frank, I'm tired of teaching the same textbook for so long though I do enjoy being a teacher. I'm considering trying something new.
```


## LeaderBoard

Descending order by Accuracy.

| Model                                                        | Accuracy | Repository                                         | Generated Text |
| ------------------------------------------------------------ | -------- | -------------------------------------------------- | -------------- |
| [Human](https://arxiv.org/abs/1902.00164)                    | $95.5$   |                                                    |                |
| [ALBERT-xxlarge + HRCA+ + Multi-Task Learning](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.651.pdf) | $92.6$   |                                                    |                |
| [ALBERT-xxlarge + DUMA + Multi-Task Learning](https://arxiv.org/abs/2003.04992) | $91.8$   |                                                    |                |
| [ALBERT-xxlarge + HRCA+](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.651.pdf) | $91.6$   |                                                    |                |
| [ALBERT-xxlarge + DUMA](https://arxiv.org/abs/2001.09415)    | $90.4$   |                                                    |                |
| [RoBERTa-Large + MMM](https://arxiv.org/abs/1910.00458)      | $88.9$   |                                                    |                |
| [XLNet-Large]()                                              | $72.0$   | [Official](https://github.com/NoviScl/XLNet_DREAM) |                |
| [BERT-Large + WAE](https://dataset.org/dream/paper/kim2020learning.pdf) | $69.0$   |                                                    |                |

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