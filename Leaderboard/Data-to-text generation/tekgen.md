# TEKGEN

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/2010.12688v2)

Repository: [Official](https://github.com/google-research-datasets/KELM-corpus)

The Dataset is part of the KELM corpus This is the Wikipedia text--Wikidata KG aligned corpus used to train the data-to-text generation model. Please note that this is a corpus generated with distant supervision and should not be used as gold standard for evaluation.

### Overview

| Dataset | Num Train   | Num Valid | Num Test  | Source Length (Avg) | Target Length (Avg) |
| ------- | ----------- | --------- | --------- | ------------------- | ------------------- |
| TEKGEN  | $6,310,061$ | $788,746$ | $796,982$ | $17.0$              | $21.2$              |

### Data Sample

Input

```
'Masi-Manimba | instance of | Town'
```

Output

```
'The town elects seven national deputies and the majority recently were from the Unified Lumumbist Party.'
```

## LeaderBoard

Descending order by BLEU.

| Model                                                | BLEU   | Repository                               | Generated Text |
| ---------------------------------------------------- | ------ | ---------------------------------------- | -------------- |
| [ReGen-SCST](https://arxiv.org/pdf/2108.12472v1.pdf) | $26.2$ | [Official](https://github.com/IBM/regen) |                |
| [ReGen-CE](https://arxiv.org/pdf/2108.12472v1.pdf)   | $24.1$ | [Official](https://github.com/IBM/regen) |                |

## Citation

```
 @inproceedings{tekgen,
    title = "Knowledge Graph Based Synthetic Corpus Generation for Knowledge-Enhanced Language Model Pre-training",
    author = "Agarwal, Oshin  and
      Ge, Heming  and
      Shakeri, Siamak  and
      Al-Rfou, Rami",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.278",
    doi = "10.18653/v1/2021.naacl-main.278",
    pages = "3554--3565",
}
```