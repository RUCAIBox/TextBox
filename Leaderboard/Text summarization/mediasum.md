# MediaSum

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/2021.naacl-main.474.pdf)

Repository: [Official](https://github.com/zcgzcgzcg1/MediaSum)

MediaSum, a large-scale media interview dataset consisting of 463.6K transcripts with abstractive summaries. To create this dataset, we collect interview transcripts from NPR and CNN and employ the overview and topic descriptions as summaries. Compared with existing public corpora for dialogue summarization, our dataset is an order of magnitude larger and contains complex multi-party conversations from multiple domains. We conduct statistical analysis to demonstrate the unique positional bias exhibited in the transcripts of televised and radioed interviews. We also show that MediaSum can be used in transfer learning to improve a model's performance on other dialogue summarization tasks.

### Overview

| Dataset  | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | -------- | ------------------- | ------------------- |
| MediaSum | $443,596$ | $10,000$  | $10,000$ | $1641.0$            | $14.4$              |

### Data Sample

```
{
  "id": "NPR-11",
  "program": "Day to Day",
  "date": "2008-06-10",
  "url": "https://www.npr.org/templates/story/story.php?storyId=91356794",
  "title": "Researchers Find Discriminating Plants",
  "summary": "The \"sea rocket\" shows preferential treatment to plants that are its kin. Evolutionary plant ecologist Susan Dudley of McMaster University in Ontario discusses her discovery.",
  "utt": [
    "This is Day to Day.  I'm Madeleine Brand.",
    "And I'm Alex Cohen.",
    "Coming up, the question of who wrote a famous religious poem turns into a very unchristian battle.",
    "First, remember the 1970s?  People talked to their houseplants, played them classical music. They were convinced plants were sensuous beings and there was that 1979 movie, \"The Secret Life of Plants.\"",
    "Only a few daring individuals, from the scientific establishment, have come forward with offers to replicate his experiments, or test his results. The great majority are content simply to condemn his efforts without taking the trouble to investigate their validity.",
    ...
    "OK. Thank you.",
    "That's Susan Dudley. She's an associate professor of biology at McMaster University in Hamilt on Ontario. She discovered that there is a social life of plants."
  ],
  "speaker": [
    "MADELEINE BRAND, host",
    "ALEX COHEN, host",
    "ALEX COHEN, host",
    "MADELEINE BRAND, host",
    "Unidentified Male",    
    ..."
    Professor SUSAN DUDLEY (Biology, McMaster University)",
    "MADELEINE BRAND, host"
  ]
}
```

## LeaderBoard

Descending order by ROUGE-2.

| Model                                            | ROUGE-1 | ROUGE-2 | ROUGE-L | Repository | Generated Text |
| ------------------------------------------------ | ------- | ------- | ------- | ---------- | -------------- |
| [BART](https://arxiv.org/pdf/2103.06410v2.pdf)   | $35.09$ | $18.05$ | $31.44$ |            |                |
| [UniLM](https://arxiv.org/pdf/2103.06410v2.pdf)  | $32.70$ | $17.27$ | $29.82$ |            |                |
| [PTGen](https://arxiv.org/pdf/2103.06410v2.pdf)  | $28.77$ | $12.24$ | $24.18$ |            |                |
| [LEAD-3](https://arxiv.org/pdf/2103.06410v2.pdf) | $14.96$ | $5.10$  | $13.29$ |            |                |

## Citation

```
 @inproceedings{zhu-etal-2021-mediasum,
    title = "{M}edia{S}um: A Large-scale Media Interview Dataset for Dialogue Summarization",
    author = "Zhu, Chenguang  and
      Liu, Yang  and
      Mei, Jie  and
      Zeng, Michael",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.474",
    doi = "10.18653/v1/2021.naacl-main.474",
    pages = "5927--5934",
}
```