# ENT-DESC

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/2004.14813)

Repository: [Official](https://github.com/LiyingCheng95/EntityDescriptionGeneration)

It is a large-scale and challenging dataset to facilitate the study of such a practical scenario in KG-to-text. The dataset involves retrieving abundant knowledge of various types of main entities from a large knowledge graph (KG), which makes the current graph-to-sequence models severely suffer from the problems of information loss and parameter explosion while generating the descriptions.

### Overview

| Dataset  | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | -------- | ------------------- | ------------------- |
| ENT-DESC | $88,652$  | $11,081$  | $11,081$ | $279.9$             | $31.0$              |

### Data Sample

Input

```
"Hatip Dicle | name in native language | Mehmet Hatip Dicle [SEP] Hatip Dicle | country of citizenship | Turkey [SEP] Hatip Dicle | member of political party | Democracy Party (Turkey) [SEP] Hatip Dicle | member of political party | People's Labor Party [SEP] Hatip Dicle | languages spoken, written or signed | Kurdish languages [SEP] Hatip Dicle | occupation | politician [SEP] Hatip Dicle | position held | Member of the Grand National Assembly of Turkey [SEP] Hatip Dicle | sex or gender | male [SEP] Hatip Dicle | educated at | Istanbul Technical University [SEP] Hatip Dicle | significant event | prisoner of conscience [SEP] Hatip Dicle | languages spoken, written or signed | Turkish [SEP] Peace and Democracy Party | country | Turkey [SEP] Hatip Dicle | date of birth | 1954-01-01 [SEP] Hatip Dicle | place of birth | Diyarbakır
```

Output

```
'Hatip Dicle (born 1954, Diyarbakir, Turkey), full name Mehmet Hatip Dicle, is a Turkish politician, of Kurdish origin, of the Peace and Democracy Party .'
```

## LeaderBoard

Descending order by BLEU.

| Model                                                | BLEU   | METEOR | TER    | ROUGE-1 | ROUGE-2 | ROUGE-L | PARENT | Repository | Generated Text |
| ---------------------------------------------------- | ------ | ------ | ------ | ------- | ------- | ------- | ------ | ---------- | -------------- |
| [MGCN](https://arxiv.org/abs/2004.14813)             | $25.7$ | $19.8$ | $69.3$ | $55.8$  | $40.0$  | $57.0$  | $23.5$ |            |                |
| [DeepGCN](https://arxiv.org/abs/2004.14813)          | $24.9$ | $19.3$ | $70.2$ | $55.0$  | $39.3$  | $56.2$  | $21.8$ |            |                |
| [GCN](https://arxiv.org/abs/2004.14813)              | $24.8$ | $19.3$ | $70.4$ | $54.9$  | $39.1$  | $56.2$  | $21.8$ |            |                |
| [GRN](https://arxiv.org/abs/2004.14813)              | $24.4$ | $18.9$ | $70.8$ | $54.1$  | $38.3$  | $55.5$  | $21.3$ |            |                |
| [GraphTransformer](https://arxiv.org/abs/2004.14813) | $19.1$ | $16.1$ | $94.5$ | $53.7$  | $37.6$  | $54.3$  | $21.4$ |            |                |
| [S2S](https://arxiv.org/abs/2004.14813)              | $6.8$  | $10.8$ | $80.9$ | $38.1$  | $21.5$  | $40.7$  | $10.0$ |            |                |

## Citation

```
 @inproceedings{ent,
    title = "{ENT}-{DESC}: Entity Description Generation by Exploring Knowledge Graph",
    author = "Cheng, Liying  and
      Wu, Dekun  and
      Bing, Lidong  and
      Zhang, Yan  and
      Jie, Zhanming  and
      Lu, Wei  and
      Si, Luo",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.90",
    doi = "10.18653/v1/2020.emnlp-main.90",
    pages = "1187--1197",
}
```