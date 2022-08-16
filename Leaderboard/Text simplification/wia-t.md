# WikiAuto+Turk/ASSET

## Dataset

### Instruction

Paper: [WikiAuto](https://aclanthology.org/2020.acl-main.709.pdf), [ASSET](https://aclanthology.org/2020.acl-main.424.pdf), [TURK](https://aclanthology.org/Q16-1029.pdf)

WikiAuto is an English simplification dataset that we paired with ASSET and TURK, two very high-quality evaluation datasets, as test sets. The input is an English sentence taken from Wikipedia and the target a simplified sentence. ASSET and TURK contain the same test examples but have references that are simplified in different ways (splitting sentences vs. rewriting and splitting).

### Overview

| Dataset             | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------------------- | --------- | --------- | -------- | ------------------- | ------------------- |
| WikiAuto+Turk/ASSET | $483,801$ | $20,000$  | $359$    |                     |                     |

### Data Sample

```
{
	'gem_id': 'wiki_auto_asset_turk-validation-0', 
	'gem_parent_id': 'wiki_auto_asset_turk-validation-0', 
	'source': 'Adjacent counties are Marin (to the south), Mendocino (to the north), Lake (northeast), Napa (to the east), and Solano and Contra Costa (to the southeast).', 
	'target': 'countries next to it are Marin, Mendocino, Lake, Napa, Solano, and Contra Costa.', 
    'references': ['countries next to it are Marin, Mendocino, Lake, Napa, Solano, and Contra Costa.']
}
```

## LeaderBoard

Descending order by METRIC.

| Model | Metric | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
|       |        |            |                |
|       |        |            |                |
|       |        |            |                |

## Citation

WikiAuto

```
 @inproceedings{jiang-etal-2020-neural,
    title = "Neural {CRF} Model for Sentence Alignment in Text Simplification",
    author = "Jiang, Chao  and
      Maddela, Mounica  and
      Lan, Wuwei  and
      Zhong, Yang  and
      Xu, Wei",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.709",
    doi = "10.18653/v1/2020.acl-main.709",
    pages = "7943--7960",
}
```

ASSET

```
@inproceedings{alva-manchego-etal-2020-asset,
    title = "{ASSET}: {A} Dataset for Tuning and Evaluation of Sentence Simplification Models with Multiple Rewriting Transformations",
    author = "Alva-Manchego, Fernando  and
      Martin, Louis  and
      Bordes, Antoine  and
      Scarton, Carolina  and
      Sagot, Beno{\^\i}t  and
      Specia, Lucia",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.424",
    pages = "4668--4679",
}
```

TURK

```
@article{Xu-EtAl:2016:TACL,
 author = {Wei Xu and Courtney Napoles and Ellie Pavlick and Quanze Chen and Chris Callison-Burch},
 title = {Optimizing Statistical Machine Translation for Text Simplification},
 journal = {Transactions of the Association for Computational Linguistics},
 volume = {4},
 year = {2016},
 url = {https://cocoxu.github.io/publications/tacl2016-smt-simplification.pdf},
 pages = {401--415}
 }
```

