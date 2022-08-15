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

```
{
    "category": "Airport",
    "lexicalisations": [
        {
            "comment": "good",
            "lex": "The leader of Aarhus is Jacob Bundsgaard.",
            "xml_id": "Id1"
        }
    ],
    "modifiedtripleset": [
        {
            "object": "Jacob_Bundsgaard",
            "property": "leaderName",
            "subject": "Aarhus"
        }
    ],
    "originaltriplesets": {
        "originaltripleset": [
            [
                {
                    "object": "Jacob_Bundsgaard",
                    "property": "leaderName",
                    "subject": "Aarhus"
                }
            ]
        ]
    },
    "shape": "",
    "shape_type": "",
    "size": "1",
    "xml_id": "Id1"
}
```

## LeaderBoard

Descending order by BLEU.

| Model                                                        | Metric  | Repository                                                | Generated Text |
| ------------------------------------------------------------ | ------- | --------------------------------------------------------- | -------------- |
| [ Control Prefixes](https://paperswithcode.com/paper/control-prefixes-for-text-generation) | $67.32$ | [Official](https://github.com/jordiclive/ControlPrefixes) |                |
| [ Multiview-G2S](https://paperswithcode.com/paper/structural-information-preserving-for-graph-1) | $62.89$ | [Official](https://github.com/Soistesimmer/AMR-multiview) |                |
| [ Graformer](https://paperswithcode.com/paper/modeling-graph-structure-via-relative) | $61.15$ |                                                           |                |

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