# WebNLG

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/P17-1017.pdf)

Repository: [Official](https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/)

The WebNLG corpus comprises of sets of triplets describing facts (entities and relations between them) and the corresponding facts in form of natural language text. The corpus contains sets with up to 7 triplets each along with one or more reference texts for each set. The test set is split into two parts: seen, containing inputs created for entities and relations belonging to DBpedia categories that were seen in the training data, and unseen, containing inputs extracted for entities and relations belonging to 5 unseen categories.

Initially, the dataset was used for the WebNLG natural language generation challenge which consists of mapping the sets of triplets to text, including referring expression generation, aggregation, lexicalization, surface realization, and sentence segmentation. The corpus is also used for a reverse task of triplets extraction.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| WebNLG  | $34,338$  | $4,313$   | $4,222$  | $18.0$              | $19.9$              |

### Data Sample

Input

```
'Abilene,_Texas | cityServed | Abilene_Regional_Airport'
```

Output

```
['Abilene, Texas is served by the Abilene regional airport.', 'Abilene Regional Airport serves the city of Abilene in Texas.']
```

## LeaderBoard

Descending order by BLEU.

| Model                                                        | BLEU    | Repository                                                | Generated Text |
| ------------------------------------------------------------ | ------- | --------------------------------------------------------- | -------------- |
| [ Control Prefixes](https://paperswithcode.com/paper/control-prefixes-for-text-generation) | $67.32$ | [Official](https://github.com/jordiclive/ControlPrefixes) |                |
| [HTML](https://arxiv.org/pdf/2107.06955v1.pdf)               | $65.4$  |                                                           |                |
| [T5-base](https://arxiv.org/pdf/2005.10433v3.pdf)            | $64.7$  |                                                           |                |
| [CGE-LW](https://arxiv.org/pdf/2001.11003v2.pdf)             | $63.69$ | [Official](https://github.com/UKPLab/kg2text)             |                |
| [ Multiview-G2S](https://paperswithcode.com/paper/structural-information-preserving-for-graph-1) | $62.89$ | [Official](https://github.com/Soistesimmer/AMR-multiview) |                |
| [ Graformer](https://paperswithcode.com/paper/modeling-graph-structure-via-relative) | $61.15$ |                                                           |                |
| [GTR-LSTM](https://aclanthology.org/P18-1151.pdf)            | $58.6$  |                                                           |                |
| [GCN EC](https://arxiv.org/pdf/1810.09995v1.pdf)             | $55.9$  | [Official](https://github.com/diegma/graph-2-text)        |                |
| [BestPlan](https://arxiv.org/pdf/1904.03396v2.pdf)           | $47.4$  | [Official](https://github.com/AmitMY/chimera)             |                |

## Citation

```
 @inproceedings{webnlg,
    title = "Creating Training Corpora for {NLG} Micro-Planners",
    author = "Gardent, Claire  and
      Shimorina, Anastasia  and
      Narayan, Shashi  and
      Perez-Beltrachini, Laura",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1017",
    doi = "10.18653/v1/P17-1017",
    pages = "179--188",
}
```