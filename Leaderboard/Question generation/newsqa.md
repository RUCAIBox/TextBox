# NewsQA

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/W17-2623.pdf)

Homepage: [Official](https://www.microsoft.com/en-us/research/project/newsqa-dataset/)

The NewsQA dataset is a crowd-sourced machine reading comprehension dataset of 120,000 question-answer pairs.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| NewsQA  | $97,850$  | $5,486$   | $5,396$  | $726.8$             | $5.0$               |

### Data Sample

Input

```
"U.S. President-elect Barack Obama [X_SEP] TEHRAN , Iran ( CNN ) -- Iran 's parliament speaker has criticized U.S. President-elect Barack Obama for saying that Iran 's development of a nuclear weapon is unacceptable . Iranian President Mahmoud Ahmadinejad has outlined where he thinks U.S. policy needs to change . Ali Larijani said Saturday that Obama should apply his campaign message of change to U.S. dealings with Iran . `` Obama must know that the change that he talks about is not simply a superficial changing of colors or tactics , '' Larijani said in comments carried by the semi-official Mehr News Agency . `` What is expected is a change in strategy , not the repetition of objections to Iran 's nuclear program , which will be taking a step in the wrong direction . '' In his first post-election news conference Friday afternoon , Obama reiterated that he believes a nuclear-armed Iran would be `` unacceptable . '' He also said he would help mount an international effort to prevent it from happening . Larijani said that U.S. behavior toward Iran `` will not change so simply '' but that Obama 's election showed internal conditions in the United States have shifted . He added that Iran does not mind if the United States provides other Persian Gulf countries with nuclear technology , but `` you should know that you can not prevent the Islamic Republic [ from reaching its goals in the nuclear field ] , '' according to the news agency . Obama cautioned Friday that it had only been a few days since the election and that he was not in office . `` Obviously , how we approach and deal with a country like Iran is not something that we should simply do in a knee-jerk fashion . I think we 've got to think it through , '' Obama said . `` But I have to reiterate once again that we only have one president at a time . And I want to be very careful that we are sending the right signals to the world as a whole that I am not the president , and I wo n't be until January 20th . '' Larijani was speaking two days after Iranian President Mahmoud Ahmadinejad congratulated Obama , the first time an Iranian leader has offered such wishes to a U.S. president-elect since the 1979 Islamic Revolution . One analyst said the welcome was a gesture from the hard-line president that he is open to a more conciliatory relationship with the United States . Ahmadinejad said Tehran `` welcomes basic and fair changes in U.S. policies and conducts , '' according to the state-run Islamic Republic News Agency on Thursday . Relations between the United States and Iran have historically been chilly and have been further strained in recent years over Iran 's nuclear program . Tehran insists that the program exists for peaceful purposes , but the United States and other Western nations are concerned by Iran 's refusal to halt uranium enrichment activities . CNN 's Shirzad Bozorgmehr contributed to this report ."
```

Output

```
'Iran criticizes who ?'
```

## LeaderBoard

Descending order by METRIC.

| Model | METRIC | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
|       |        |            |                |
|       |        |            |                |
|       |        |            |                |

## Citation

```
@inproceedings{trischler-etal-2017-newsqa,
    title = "{N}ews{QA}: A Machine Comprehension Dataset",
    author = "Trischler, Adam  and
      Wang, Tong  and
      Yuan, Xingdi  and
      Harris, Justin  and
      Sordoni, Alessandro  and
      Bachman, Philip  and
      Suleman, Kaheer",
    booktitle = "Proceedings of the 2nd Workshop on Representation Learning for {NLP}",
    month = aug,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-2623",
    doi = "10.18653/v1/W17-2623",
    pages = "191--200",
}
```

