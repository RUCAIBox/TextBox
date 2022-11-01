# E2E

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1706.09254)

Repository: [Official](http://www.macs.hw.ac.uk/InteractionLab/E2E/)

It is a new dataset for training end-to-end, data-driven natural language generation systems in the restaurant domain, which is ten times bigger than existing, frequently used datasets in this area. The E2E dataset poses new challenges: (1) its human reference texts show more lexical richness and syntactic variation, including discourse phenomena; (2) generating from this set requires content selection. As such, learning from this dataset promises more natural, varied and less template-like system utterances.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| E2E     | $42,061$  | $4,672$   | $4,693$  | $25.7$              | $20.3$              |

### Data Sample

Input
```
name[The Eagle],
eatType[coffee shop],
food[French],
priceRange[moderate],
customerRating[3/5],
area[riverside],
kidsFriendly[yes],
near[Burger King]
```
Output
```
The three star coffee shop, The Eagle, gives families a mid-priced dining experience featuring a variety of wines and cheeses. Find The Eagle near Burger King.
```
## LeaderBoard

Descending order by BLEU.

| Model                                                 | BLEU    | NIST     | METEOR  | ROUGE-L | CIDEr    | Repository                                                   | Generated Text |
| ----------------------------------------------------- | ------- | -------- | ------- | ------- | -------- | ------------------------------------------------------------ | -------------- |
| [HTLM](https://arxiv.org/pdf/2107.06955v1.pdf)        | $70.3$  | $8.90$   | $46.3$  | $70.8$  | $2.47$   |                                                              |                |
| [$S_1^R$](https://arxiv.org/pdf/1904.01301v2.pdf)     | $68.60$ | $8.73$   | $45.25$ | $70.82$ | $2.37$   |                                                              |                |
| [GPT-2-Large](https://arxiv.org/pdf/2107.06955v1.pdf) | $68.5$  | $8.78$   | $46.0$  | $69.9$  | $2.45$   |                                                              |                |
| [EDA_CS](https://arxiv.org/pdf/1904.11838v4.pdf)      | $67.05$ | $8.5150$ | $44.49$ | $68.94$ | $2.2355$ | [Official](https://github.com/marco-roberti/char-data-to-text-gen) |                |
| [Slug](https://arxiv.org/pdf/1805.06553v1.pdf) | $66.19$ | $8.6130$ | $44.54$ | $67.72$ |  |  | |
| [TGen](https://arxiv.org/pdf/1810.01170v1.pdf) | $65.93$ | $8.6094$ | $44.83$ | $68.50$ | $2.2338$ | [Official](https://github.com/UFAL-DSG/tgen) ||

## Citation

```
 @inproceedings{e2e,
    title = "The {E}2{E} Dataset: New Challenges For End-to-End Generation",
    author = "Novikova, Jekaterina  and
      Du{\v{s}}ek, Ond{\v{r}}ej  and
      Rieser, Verena",
    booktitle = "Proceedings of the 18th Annual {SIG}dial Meeting on Discourse and Dialogue",
    month = aug,
    year = "2017",
    address = {Saarbr{\"u}cken, Germany},
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-5525",
    doi = "10.18653/v1/W17-5525",
    pages = "201--206",
}
```