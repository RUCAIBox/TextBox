# GenWiki

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/2020.coling-main.217.pdf)

Repository: [Official](https://github.com/zhijing-jin/genwiki)

GenWiki is a large-scale dataset for *knowledge graph-to-text* (G2T) and *text-to-knowledge graph* (T2G) conversion.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| GenWiki | $681,436$ | $75,716$  | $1,000$  | $21.4$              | $29.5$              |

### Data Sample

```
{
    "text": "It has been <ENT_0> the permanent collection of the <ENT_1> <ENT_0> <ENT_2> since <ENT_4> , acquired through the <ENT_3> .",
    "entities": [
        "in",
        "Museum of Modern Art",
        "New York City",
        "Lillie P. Bliss Bequest",
        "1941"
    ],
    "graph": [
        [
            "The Starry Night",
            "city",
            "New York City"
        ]
    ],
    "id_long": {
        "wikipage": "The_Starry_Night",
        "text_paragraph_index": 0,
        "text_sentence_index_start": 2,
        "text_sentence_index_end": 3,
        "graph_set_index": 0
    },
    "id_short": "[\"The_Starry_Night\", 0, [0, 2, 3]]"
}
```

## LeaderBoard

Descending order by BLEU.

| Model                                                        | BLEU    | METEOR  | ROUGE-L | CIDEr  | Repository | Generated Text |
| ------------------------------------------------------------ | ------- | ------- | ------- | ------ | ---------- | -------------- |
| [CycleGT](https://aclanthology.org/2020.coling-main.217.pdf) | $41.29$ | $35.39$ | $63.73$ | $3.53$ |            |                |
| [Graph Transformer Model](https://aclanthology.org/2020.coling-main.217.pdf) | $35.03$ | $33.45$ | $58.14$ | $2.63$ |            |                |

## Citation

```
 @inproceedings{genwiki,
    title = "{G}en{W}iki: A Dataset of 1.3 Million Content-Sharing Text and Graphs for Unsupervised Graph-to-Text Generation",
    author = "Jin, Zhijing  and
      Guo, Qipeng  and
      Qiu, Xipeng  and
      Zhang, Zheng",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.217",
    doi = "10.18653/v1/2020.coling-main.217",
    pages = "2398--2409",
}
```