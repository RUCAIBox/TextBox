# ToTTo

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/2004.14373)

Repository: [Official](https://github.com/google-research-datasets/ToTTo)

It is an open-domain English table-to-text dataset with over 120,000 training examples that proposes a controlled generation task: given a Wikipedia table and a set of highlighted table cells, produce a one-sentence description. 

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| ToTTo   | $120,761$ | $7,700$   | $7,700$  | $32.5$              | $14.8$              |

### Data Sample

```
{
  "table_page_title": "'Weird Al' Yankovic",
  "table_webpage_url": "https://en.wikipedia.org/wiki/%22Weird_Al%22_Yankovic",
  "table_section_title": "Television",
  "table_section_text": "",
  "table": "[Described below]",
  "highlighted_cells": [[22, 2], [22, 3], [22, 0], [22, 1], [23, 3], [23, 1], [23, 0]],
  "example_id": 12345678912345678912,
  "sentence_annotations": [{"original_sentence": "In 2016, Al appeared in 2 episodes of BoJack Horseman as Mr. Peanutbutter's brother, Captain Peanutbutter, and was hired to voice the lead role in the 2016 Disney XD series Milo Murphy's Law.",
                  "sentence_after_deletion": "In 2016, Al appeared in 2 episodes of BoJack Horseman as Captain Peanutbutter, and was hired to the lead role in the 2016 series Milo Murphy's Law.",
                  "sentence_after_ambiguity": "In 2016, Al appeared in 2 episodes of BoJack Horseman as Captain Peanutbutter, and was hired for the lead role in the 2016 series Milo Murphy's 'Law.",
                  "final_sentence": "In 2016, Al appeared in 2 episodes of BoJack Horseman as Captain Peanutbutter and was hired for the lead role in the 2016 series Milo Murphy's Law."}],
}
```

## LeaderBoard

Descending order by BLEU.

| Model                                                 | BLEU   | PARENT | Repository | Generated Text |
| ----------------------------------------------------- | ------ | ------ | ---------- | -------------- |
| [BERT-to-BERT](https://arxiv.org/abs/2004.14373)      | $44.0$ | $52.6$ |            |                |
| [Pointer-Generator](https://arxiv.org/abs/2004.14373) | $41.6$ | $51.6$ |            |                |

## Citation

```
 @inproceedings{totto2,
    title = "{ToTTo}: A Controlled Table-To-Text Generation Dataset",
    author = "Parikh, Ankur  and
      Wang, Xuezhi  and
      Gehrmann, Sebastian  and
      Faruqui, Manaal  and
      Dhingra, Bhuwan  and
      Yang, Diyi  and
      Das, Dipanjan",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.89",
    doi = "10.18653/v1/2020.emnlp-main.89",
    pages = "1173--1186",
}
```