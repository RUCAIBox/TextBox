# MediaSum

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/2021.naacl-main.474.pdf)

Repository: [Official](https://github.com/zcgzcgzcg1/MediaSum)

MediaSum, a large-scale media interview dataset consisting of 463.6K transcripts with abstractive summaries. To create this dataset, we collect interview transcripts from NPR and CNN and employ the overview and topic descriptions as summaries. Compared with existing public corpora for dialogue summarization, our dataset is an order of magnitude larger and contains complex multi-party conversations from multiple domains. We conduct statistical analysis to demonstrate the unique positional bias exhibited in the transcripts of televised and radioed interviews. We also show that MediaSum can be used in transfer learning to improve a model's performance on other dialogue summarization tasks.

### Overview

| Dataset  | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | -------- | ------------------- | ------------------- |
| MediaSum | $443,596$ | $10,000$  | $10,000$ | $1641.0$            | $14.4$              |

### Data Sample

Input

```
'RACHEL MARTIN, HOST: Good morning. I\'m Rachel Martin. A middle school art teacher in Abilene, Texas, wanted to reward her hardworking students. And what could ease their stress and spark their creativity? Dressing up like Bob Ross, of course. On the designated day, dozens of eighth graders wore big, curly, reddish wigs and chambray shirts. Then they painted along to an episode of Ross\' PBS show "The Joy Of Painting." The Abilene Reporter-News said it best. It was a Bob Ross flash mob - a flash Bob, if you will. It\'s MORNING EDITION.'
```

Output

```
'What could ease their stress? Dressing up like Bob Ross. Dozens of eighth graders wore big, curly wigs and chambray shirts, and then painted along to an episode of his PBS show The Joy of Painting.'
```

## LeaderBoard

Descending order by ROUGE-2.

| Model                                            | ROUGE-1 | ROUGE-2 | ROUGE-L | Repository | Generated Text |
| ------------------------------------------------ | ------- | ------- | ------- | ---------- | -------------- |
| [BART](https://arxiv.org/pdf/2103.06410v2.pdf)   | $35.09$ | $18.05$ | $31.44$ |            |                |
| [UniLM](https://arxiv.org/pdf/2103.06410v2.pdf)  | $32.70$ | $17.27$ | $29.82$ |            |                |
| [PTGen](https://arxiv.org/pdf/2103.06410v2.pdf)  | $28.77$ | $12.24$ | $24.18$ |            |                |
| [LEAD-3](https://arxiv.org/pdf/2103.06410v2.pdf) | $14.96$ | $5.10$  | $13.29$ |            |                |

## Citation

```
 @inproceedings{zhu-etal-2021-mediasum,
    title = "{M}edia{S}um: A Large-scale Media Interview Dataset for Dialogue Summarization",
    author = "Zhu, Chenguang  and
      Liu, Yang  and
      Mei, Jie  and
      Zeng, Michael",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.474",
    doi = "10.18653/v1/2021.naacl-main.474",
    pages = "5927--5934",
}
```