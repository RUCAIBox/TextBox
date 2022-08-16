# TriviaQA

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/P17-1147.pdf)

Homepage: [Official](http://nlp.cs.washington.edu/triviaqa/)

TriviaQA is a realistic text-based question answering dataset which includes 950K question-answer pairs from 662K documents collected from Wikipedia and the web. This dataset is more challenging than standard QA benchmark datasets such as Stanford Question Answering Dataset (SQuAD), as the answers for a question may not be directly obtained by span prediction and the context is very long. TriviaQA dataset consists of both human-verified and machine-generated QA subsets.

### Overview

| Dataset  | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | -------- | ------------------- | ------------------- |
| TriviaQA | $78,785$  | $8,837$   | $11,313$ | $14.0$              | $2.0$               |

### Data Sample

Input

```

```

Output

```

```

## LeaderBoard

Descending order by EM.

| Model                                                  | EM     | Repository                                             | Generated Text |
| ------------------------------------------------------ | ------ | ------------------------------------------------------ | -------------- |
| [ PaLM-540B](https://arxiv.org/pdf/2204.02311v3.pdf)   | $81.4$ | [Official](https://github.com/lucidrains/PaLM-pytorch) |                |
| [GLaM 62B/64E](https://arxiv.org/pdf/2112.06905v2.pdf) | $75.8$ |                                                        |                |
| [ FiD+Distil](https://arxiv.org/pdf/2012.04584v2.pdf)  | $72.1$ | [Official](https://github.com/facebookresearch/FiD)    |                |

## Citation

```
@inproceedings{joshi-etal-2017-triviaqa,
    title = "{T}rivia{QA}: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension",
    author = "Joshi, Mandar  and
      Choi, Eunsol  and
      Weld, Daniel  and
      Zettlemoyer, Luke",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1147",
    doi = "10.18653/v1/P17-1147",
    pages = "1601--1611",
}
```

