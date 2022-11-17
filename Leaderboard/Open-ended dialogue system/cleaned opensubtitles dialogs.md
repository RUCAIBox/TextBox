# Cleaned OpenSubtitles Dialogs

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/L18-1275.pdf)

OpenSubtitles is collection of multilingual parallel corpora. The dataset is compiled from a large database of movie and TV subtitles and includes a total of 1689 bitexts spanning 2.6 billion sentences across 60 languages.

### Overview

| Dataset                       | Num Train    | Num Valid   | Num Test | Source Length (Avg) | Target Length (Avg) |
| ----------------------------- | ------------ | ----------- | -------- | ------------------- | ------------------- |
| Cleaned OpenSubtitles Dialogs | $13,355,487$ | $1,483,944$ | -        | $75.5$              | $16.7$              |

### Data Sample

Input

```
'Have you reached a verdict ? [SEP] We have , your honor .'
```

Output

```
'What is your verdict ?'
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
 @inproceedings{lison-etal-2018-opensubtitles2018,
    title = "{O}pen{S}ubtitles2018: Statistical Rescoring of Sentence Alignments in Large, Noisy Parallel Corpora",
    author = {Lison, Pierre  and
      Tiedemann, J{\"o}rg  and
      Kouylekov, Milen},
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1275",
}
```