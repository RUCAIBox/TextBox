# Frames

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/W17-5526v2.pdf)

Homepage: [Official](https://www.microsoft.com/en-us/research/project/frames-dataset/)

This dataset is dialog dataset collected in a Wizard-of-Oz fashion. Two humans talked to each other via a chat interface. One was playing the role of the user and the other one was playing the role of the conversational agent. The latter is called a wizard as a reference to the Wizard of Oz, the man behind the curtain. The wizards had access to a database of 250+ packages, each composed of a hotel and round-trip flights. The users were asked to find the best deal. This resulted in complex dialogues where a user would often consider different options, compare packages, and progressively build the description of her ideal trip.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| Frames  | $26,631$  | $2,106$   | -        | $116.1$             | $13.0$              |

### Data Sample

Input

```
'Belief state [X_SEP] Hello bot. I want to go to Manas from Boston on September 2'
```

Output

```
'[booking] intent book destination city Manas origin city Boston start date September 2'
```

## LeaderBoard

Descending order by METRIC.

| Model | Metric | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
|       |        |            |                |
|       |        |            |                |
|       |        |            |                |

## Citation

```
 @inproceedings{el-asri-etal-2017-frames,
    title = "{F}rames: a corpus for adding memory to goal-oriented dialogue systems",
    author = "El Asri, Layla  and
      Schulz, Hannes  and
      Sharma, Shikhar  and
      Zumer, Jeremie  and
      Harris, Justin  and
      Fine, Emery  and
      Mehrotra, Rahul  and
      Suleman, Kaheer",
    booktitle = "Proceedings of the 18th Annual {SIG}dial Meeting on Discourse and Dialogue",
    month = aug,
    year = "2017",
    address = {Saarbr{\"u}cken, Germany},
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-5526",
    doi = "10.18653/v1/W17-5526",
    pages = "207--219",
}
```