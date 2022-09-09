# WebQuestions

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/D13-1160.pdf)

The **WebQuestions** dataset is a question answering dataset using Freebase as the knowledge base and contains 6,642 question-answer pairs. It was created by crawling questions through the Google Suggest API, and then obtaining answers using Amazon Mechanical Turk. The original split uses 3,778 examples for training and 2,032 for testing. All answers are defined as Freebase entities.

### Overview

| Dataset      | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------------ | --------- | --------- | -------- | ------------------- | ------------------- |
| WebQuestions | $8,933$   | $4,863$   | $4,863$  | $6.7$               | $2.4$               |

### Data Sample

```
{
    "answers": ["Jamaican Creole English Language", "Jamaican English"],
    "question": "what does jamaican people speak?",
    "url": "http://www.freebase.com/view/en/jamaica"
}
```

## LeaderBoard

Descending order by EM.

| Model                                                | EM     | Repository                                             | Generated Text |
| ---------------------------------------------------- | ------ | ------------------------------------------------------ | -------------- |
| [ RAG](https://arxiv.org/pdf/2005.11401v4.pdf)       | $45.2$ |                                                        |                |
| [ PaLM-540B](https://arxiv.org/pdf/2204.02311v3.pdf) | $43.5$ | [Official](https://github.com/lucidrains/PaLM-pytorch) |                |
| [DPR](https://arxiv.org/pdf/2004.04906v3.pdf)        | $42.4$ | [Official](https://github.com/facebookresearch/DPR)    |                |

## Citation

```
@inproceedings{berant-etal-2013-semantic,
    title = "Semantic Parsing on {F}reebase from Question-Answer Pairs",
    author = "Berant, Jonathan  and
      Chou, Andrew  and
      Frostig, Roy  and
      Liang, Percy",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D13-1160",
    pages = "1533--1544",
}
```

