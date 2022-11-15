# Topical-Chat

## Dataset

### Instruction

Paper: [Paper](https://www.isca-speech.org/archive/interspeech_2019/gopalakrishnan19_interspeech.html)

Repository: [Official](https://github.com/alexa/Topical-Chat)

A knowledge-grounded human-human conversation dataset where the underlying knowledge spans 8 broad topics and conversation partners don’t have explicitly defined roles.

### Overview

| Dataset      | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------------ | --------- | --------- | -------- | ------------------- | ------------------- |
| Topical-Chat | $179,750$ | $22,295$  | $22,452$ | $223.3$             | $20.0$              |

### Data Sample

Input

```
"Are you a football fan? [SEP] yes, I'm. a football fan. [SEP] Did you now that the university of Iowa's visiting football teams locker room is painted pink?"
```

Output

```
'Yes, I do. I think they want to intimidate the visiting teams.'
```

## LeaderBoard

Descending order by Spearman Correlation.

| Model                                                       | Spearman Correlation | Pearson Correlation | Repository                                                   | Generated Text |
| ----------------------------------------------------------- | -------------------- | ------------------- | ------------------------------------------------------------ | -------------- |
| [MDD-Eval](https://arxiv.org/pdf/2112.07194v2.pdf)          | $51.09$              | $45.75$             | [Offcial](https://github.com/e0397123/mdd-eval)              |                |
| [Lin-Reg](https://aclanthology.org/2021.emnlp-main.618.pdf) | $48.77$              | $49.74$             | [Official](https://github.com/smartdataanalytics/proxy_indicators) |                |
| [USR](https://arxiv.org/pdf/2005.00456v1.pdf)               | $32.45$              | $40.68$             | [Official](https://github.com/shikib/usr)                    |                |

## Citation

```
 @inproceedings{tc,
  author    = {Karthik Gopalakrishnan and
               Behnam Hedayatnia and
               Qinglang Chen and
               Anna Gottardi and
               Sanjeev Kwatra and
               Anu Venkatesh and
               Raefer Gabriel and
               Dilek Hakkani{-}T{\"{u}}r},
  title     = {Topical-Chat: Towards Knowledge-Grounded Open-Domain Conversations},
  booktitle = {Interspeech 2019, 20th Annual Conference of the International Speech
               Communication Association},
  pages     = {1891--1895},
  publisher = {{ISCA}},
  year      = {2019},
  url={https://doi.org/10.21437/Interspeech.2019-3079},
  doi       = {10.21437/Interspeech.2019-3079},
}
```