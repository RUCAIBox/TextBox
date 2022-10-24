# GYAFC-E&M/F&R

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/N18-1012.pdf)

Repository: [Official](https://github.com/raosudha89/GYAFC-corpus)

Grammarly’s Yahoo Answers Formality Corpus (GYAFC) is the largest dataset for any style containing a total of 110K informal / formal sentence pairs.

Yahoo Answers is a question answering forum, contains a large number of informal sentences and allows redistribution of data. The authors used the Yahoo Answers L6 corpus to create the GYAFC dataset of informal and formal sentence pairs. In order to ensure a uniform distribution of data, they removed sentences that are questions, contain URLs, and are shorter than 5 words or longer than 25. After these preprocessing steps, 40 million sentences remain.

The Yahoo Answers corpus consists of several different domains like Business, Entertainment & Music, Travel, Food, etc. Pavlick and Tetreault formality classifier (PT16) shows that the formality level varies significantly across different genres. In order to control for this variation, the authors work with two specific domains that contain the most informal sentences and show results on training and testing within those categories. The authors use the formality classifier from PT16 to identify informal sentences and train this classifier on the Answers genre of the PT16 corpus which consists of nearly 5,000 randomly selected sentences from Yahoo Answers manually annotated on a scale of -3 (very informal) to 3 (very formal).  They find that the domains of Entertainment & Music and Family & Relationships contain the most informal sentences and create the GYAFC dataset using these domains.

### Overview

| Dataset   | Num Train | Num Test(Informal to Formal) | Num Test(Formal to Informal) |
| --------- | --------- | ---------------------------- | ---------------------------- |
| GYAFC-E&M | $52,595$  | $1,416$                      | $1,082$                      |
| GYAFC-F&R | $51,967$  | $1,332$                      | $1,019$                      |

### Data Sample

#### GYAFC-E&M

Input

```
Is Any Baby Really A Freak.
```

Output

```
['Is any baby really a freak.', 'Is any baby really a freak?', 'Is Any Baby really strange?', 'Is anyone really that crazy?']
```

#### GYAFC-F&R

Input

```
And so what if it is a rebound relationship for both of you?
```

Output

```
['What if it is a rebound relationship for both of you?', 'Is it a rebound relationship for both of you?', 'So what if it is a rebound relationship for the two of you?', 'What does it matter if it is a rebound relationship for you both?'] 
```

## LeaderBoard

Descending order by BLEU.

| Model                                                | BLEU   | Repository                                                   | Generated Text |
| ---------------------------------------------------- | ------ | ------------------------------------------------------------ | -------------- |
| [DualRL](https://arxiv.org/pdf/1905.10060v1.pdf)     | $41.9$ | [Official](https://github.com/luofuli/DualRL)                |                |
| [Template](https://arxiv.org/pdf/1804.06437v1.pdf)   | $35.2$ | [Official](https://github.com/lijuncen/Sentiment-and-Style-Transfer) |                |
| [UnsuperMT](https://arxiv.org/pdf/1808.07894v1.pdf)  | $33.4$ |                                                              |                |
| [Del](https://arxiv.org/pdf/1804.06437v1.pdf)        | $29.2$ | [Official](https://github.com/lijuncen/Sentiment-and-Style-Transfer) |                |
| [DelRetri](https://arxiv.org/pdf/1804.06437v1.pdf)   | $21.2$ | [Official](https://github.com/lijuncen/Sentiment-and-Style-Transfer) |                |
| [MultiDec](https://arxiv.org/pdf/1711.06861v2.pdf)   | $12.3$ |                                                              |                |
| [StyleEmbed](https://arxiv.org/pdf/1711.06861v2.pdf) | $7.9$  |                                                              |                |
| [CrossAlign](https://arxiv.org/pdf/1705.09655v2.pdf) | $3.6$  | [Official](https://github.com/shentianxiao/language-style-transfer) |                |
| [Unpair](https://arxiv.org/pdf/1805.05181v2.pdf)     | $2.0$  | [Official](https://github.com/lancopku/unpaired-sentiment-translation) |                |
| [BackTrans](https://arxiv.org/pdf/1804.09000v3.pdf)  | $0.9$  | [Official](https://github.com/shrimai/Style-Transfer-Through-Back-Translation) |                |
| [Retri](https://arxiv.org/pdf/1804.06437v1.pdf)      | $0.4$  | [Official](https://github.com/lijuncen/Sentiment-and-Style-Transfer) |                |

## Citation

```
 @inproceedings{rao-tetreault-2018-dear,
    title = "Dear Sir or Madam, May {I} Introduce the {GYAFC} Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer",
    author = "Rao, Sudha  and
      Tetreault, Joel",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-1012",
    doi = "10.18653/v1/N18-1012",
    pages = "129--140",
}
```