# WikiAuto+Turk/ASSET

## Dataset

### Instruction

Paper: [WikiAuto](https://aclanthology.org/2020.acl-main.709.pdf), [ASSET](https://aclanthology.org/2020.acl-main.424.pdf), [TURK](https://aclanthology.org/Q16-1029.pdf)

Repository: [WikiAuto](https://github.com/chaojiang06/wiki-auto)

WikiAuto is an English simplification dataset that we paired with ASSET and TURK, two very high-quality evaluation datasets, as test sets. The input is an English sentence taken from Wikipedia and the target a simplified sentence. ASSET and TURK contain the same test examples but have references that are simplified in different ways (splitting sentences vs. rewriting and splitting).

### Overview

| Dataset             | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------------------- | --------- | --------- | -------- | ------------------- | ------------------- |
| WikiAuto+Turk/ASSET | $483,801$ | $359$     | $359$    | $26.2$              | $21.5$              |

### Data Sample

Input

```
'One side of the armed conflicts is made of Sudanese military and the Janjaweed, a Sudanese militia recruited from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan.'
```

Output

```
['One side of the armed conflicts is composed mainly of the Sudanese military and the Janjaweed, a Sudanese militia group recruited mostly from the Afro-Arab Abbala tribes of the northern Rizeigat regime in Sudan.', 'One side of the armed conflicts is made up mostly of the Sudanese military and the Janjaweed, a Sudanese militia group whose recruits mostly come from the Afro-Arab Abbala tribes from the northern Rizeigat region in Sudan.', 'One side of the armed conflicts is composed mainly of the Sudanese military and the Janjaweed, a Sudanese militia group recruited mostly from the Afro-Arab Abbala tribes in Sudan.', 'One side of the armed conflicts is composed mainly of the Sudanese military and the Janjaweed, a Sudanese militia group recruited mostly from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan.', 'One side of the armed conflicts consist of the Sudanese military and the Sudanese militia group Janjaweed.', 'The Sudanese military and the Janjaweed make up one of the armed conflicts, mostly from the Afro-Arab Abbal tribes in Sudan.', 'One side of the armed conflicts is mainly Sudanese military and the Janjaweed, which recruited from the Afro-Arab Abbala tribes.']
```

## LeaderBoard

#### Turk

Descending order by SARI.

| Model                                                    | SARI    | Repository                                                  | Generated Text |
| -------------------------------------------------------- | ------- | ----------------------------------------------------------- | -------------- |
| [MUSS](https://arxiv.org/pdf/2005.00352v2.pdf)           | $42.53$ | [Official](https://github.com/facebookresearch/muss)        |                |
| [Control Prefix](https://arxiv.org/pdf/2110.08329v2.pdf) | $42.32$ | [Official](https://github.com/jordiclive/ControlPrefixes)   |                |
| [TST](https://arxiv.org/pdf/2103.05070v1.pdf)            | $41.46$ | [Official](https://github.com/grammarly/gector)             |                |
| [ACCESS](https://arxiv.org/pdf/1910.02677v3.pdf)         | $41.38$ | [Official](https://github.com/facebookresearch/access)      |                |
| [DMASS-DCSS](https://arxiv.org/pdf/1810.11193v1.pdf)     | $40.45$ | [Official](https://github.com/Sanqiang/text_simplification) |                |
| [EditNTS](https://arxiv.org/pdf/1906.08104v1.pdf)        | $38.22$ | [Official](https://github.com/YueDongCS/EditNTS)            |                |
| [Edit-Unsup-TS](https://arxiv.org/pdf/2006.09639v1.pdf)  | $37.85$ | [Official](https://github.com/ddhruvkr/Edit-Unsup-TS)       |                |

#### ASSET

Descending order by SARI.

| Model                                                    | SARI    | Repository                                                  | Generated Text |
| -------------------------------------------------------- | ------- | ----------------------------------------------------------- | -------------- |
| [MUSS](https://arxiv.org/pdf/2005.00352v2.pdf)           | $44.15$ | [Official](https://github.com/facebookresearch/muss)        |                |
| [Control Prefix](https://arxiv.org/pdf/2110.08329v2.pdf) | $43.58$ | [Official](https://github.com/jordiclive/ControlPrefixes)   |                |
| [TST](https://arxiv.org/pdf/2103.05070v1.pdf)            | $43.21$ | [Official](https://github.com/grammarly/gector)             |                |
| [ACCESS](https://arxiv.org/pdf/1910.02677v3.pdf)         | $40.13$ | [Official](https://github.com/facebookresearch/access)      |                |
| [DMASS-DCSS](https://arxiv.org/pdf/1810.11193v1.pdf)     | $38.67$ | [Official](https://github.com/Sanqiang/text_simplification) |                |
| [Dress-LS](https://arxiv.org/pdf/1703.10931v2.pdf)       | $36.59$ |                                                             |                |
| [UNTS](https://arxiv.org/pdf/1810.07931v6.pdf)           | $35.19$ | [Official](https://github.com/subramanyamdvss/UnsupNTS)     |                |
| [PBMT-R](https://aclanthology.org/P12-1107.pdf)          | $34.63$ |                                                             |                |

## Citation

WikiAuto

```
@inproceedings{jiang2020neural,
  title={Neural CRF Model for Sentence Alignment in Text Simplification},
  author={Jiang, Chao and Maddela, Mounica and Lan, Wuwei and Zhong, Yang and Xu, Wei},
  booktitle={Proceedings of the Association for Computational Linguistics (ACL)},
  year={2020}
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

