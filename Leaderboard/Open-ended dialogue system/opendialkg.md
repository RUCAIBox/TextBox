# OpenDialKG

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/P19-1081.pdf)

Repository: [Official](https://github.com/facebookresearch/opendialkg)

OpenDialKG contains utterance from 15K human-to-human role-playing dialogs is manually annotated with ground-truth reference to corresponding entities and paths from a large-scale KG with 1M+ facts.

### Overview

| Dataset    | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ---------- | --------- | --------- | -------- | ------------------- | ------------------- |
| OpenDialKG | $69,680$  | $7,743$   | -        | $54.2$              | $12.4$              |

### Data Sample

Input

```
"I think Rhianna is one of the best singers of our generation, what do you think? [SEP] I really don't listen to her music I did see her in Battleship though [SEP] I have never seen her act, is she any good? [SEP] She wasn't to bad Petter Berg was also in Battleship with her [SEP] I should watch that, I like Liam Neeson. Also I always get those Skarsgard brothers mixed up. [SEP] Me to I know that Liam Neeson starred in Wrath of the Titans"
```

Output

```
'Yes, I saw that one. I thought he was great in Kinsey as well which is probably one of his more underrated roles.'
```

## LeaderBoard

Descending order by R@1.

| Model                                                  | R@1    | R@3    | R@5    | R@10   | R@25   | Repository | Generated Text |
| ------------------------------------------------------ | ------ | ------ | ------ | ------ | ------ | ---------- | -------------- |
| [DialKG Walker](https://aclanthology.org/P19-1081.pdf) | $13.2$ | $26.1$ | $35.3$ | $47.9$ | $62.2$ |            |                |
| [Tri-LSTM](https://aclanthology.org/P19-1081.pdf)      | $3.2$  | $14.2$ | $22.6$ | $36.3$ | $56.2$ |            |                |
| [Seq2Seq](https://aclanthology.org/P19-1081.pdf)       | $3.1$  | $18.3$ | $29.7$ | $44.1$ | $60.2$ |            |                |
| [Ext-ED](https://aclanthology.org/P19-1081.pdf)        | $1.9$  | $5.8$  | $9.0$  | $13.3$ | $19.0$ |            |                |

## Citation

```
 @inproceedings{moon-etal-2019-opendialkg,
    title = "{O}pen{D}ial{KG}: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs",
    author = "Moon, Seungwhan  and
      Shah, Pararth  and
      Kumar, Anuj  and
      Subba, Rajen",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1081",
    doi = "10.18653/v1/P19-1081",
    pages = "845--854",
}
```