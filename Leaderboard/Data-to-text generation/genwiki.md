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

Input

```
'White Coppice | country | England [SEP] White Coppice | district | Chorley [SEP] White Coppice | county | Lancashire [SEP] White Coppice | settlement Type | hamlet'
```

Output

```
'White Coppice is a hamlet near Chorley , Lancashire , England .'
```

## LeaderBoard

Descending order by BLEU.

| Model                                                        | BLEU    | METEOR  | ROUGE-L | CIDEr  | Repository | Generated Text |
| ------------------------------------------------------------ | ------- | ------- | ------- | ------ | ---------- | -------------- |
| [CycleGT_Base](https://aclanthology.org/2020.coling-main.217.pdf) | $41.29$ | $35.39$ | $63.73$ | $3.53$ |            |                |
| [CycleGT_Warm]((https://aclanthology.org/2020.coling-main.217.pdf)) | $40.47$ | $34.84$ | $63.40$ | $3.48$ |            |                |
| [Graph Transformer Model](https://aclanthology.org/2020.coling-main.217.pdf) | $35.03$ | $33.45$ | $58.14$ | $2.63$ |            |                |
| [DirectTransfer]((https://aclanthology.org/2020.coling-main.217.pdf)) | $13.89$ | $25.76$ | $39.75$ | $1.26$ |            |                |
| [Rule-Based](https://aclanthology.org/2020.coling-main.217.pdf) | $13.45$ | $30.72$ | $40.93$ | $1.26$ |            |                |

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