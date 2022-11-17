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
'Who was the man behind The Chipmunks?'
```

Output

```
'David Seville'
```

## LeaderBoard

Descending order by F1.

| Model                                                        | EM     | F1     | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | ------ | ------ | ------------------------------------------------------------ | -------------- |
| [BigBird-etc](https://arxiv.org/pdf/2007.14062v2.pdf)        | -      | $80.9$ | [Official](https://github.com/google-research/bigbird)       |                |
| [LinkBERT](https://arxiv.org/pdf/2203.15827v1.pdf)           | -      | $78.2$ | [Official](https://github.com/michiyasunaga/LinkBERT)        |                |
| [ UnitedQA](https://arxiv.org/pdf/2101.00178v2.pdf)          | -      | $70.3$ |                                                              |                |
| [ ReasonBERTR](https://arxiv.org/pdf/2109.04912v1.pdf)       | -      | $45.5$ | [Official](https://github.com/sunlab-osu/reasonbert)         |                |
| [ PaLM-540B](https://arxiv.org/pdf/2204.02311v3.pdf)         | $81.4$ | -      | [Official](https://github.com/lucidrains/PaLM-pytorch)       |                |
| [GLaM 62B/64E](https://arxiv.org/pdf/2112.06905v2.pdf)       | $75.8$ | -      |                                                              |                |
| [ FiD+Distil](https://arxiv.org/pdf/2012.04584v2.pdf)        | $72.1$ | -      | [Official](https://github.com/facebookresearch/FiD)          |                |
| [EMDR2](https://arxiv.org/pdf/2106.05346v2.pdf)              | $71.4$ | -      | [Official](https://github.com/DevSinghSachan/emdr2)          |                |
| [GPT-3 175B](https://arxiv.org/pdf/2005.14165v4.pdf)         | $71.2$ | -      | [Official](https://github.com/openai/gpt-3)                  |                |
| [Fusion-in-Decoder](https://arxiv.org/pdf/2007.01282v2.pdf)  | $67.6$ | -      |                                                              |                |
| [TOME-2](https://arxiv.org/pdf/2110.06176v2.pdf)             | $65.8$ | -      | [Official](https://github.com/google-research/language/tree/master/language/mentionmemory) |                |
| [FLAN 137B zero-shot](https://arxiv.org/pdf/2109.01652v5.pdf) | $56.7$ | -      | [Official](https://github.com/google-research/flan)          |                |
| [RAG](https://arxiv.org/pdf/2005.11401v4.pdf)                | $56.1$ | -      |                                                              |                |


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

