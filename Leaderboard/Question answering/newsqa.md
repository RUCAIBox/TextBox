# NewsQA

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/W17-2623.pdf)

Homepage: [Official](https://www.microsoft.com/en-us/research/project/newsqa-dataset/)

The NewsQA dataset is a crowd-sourced machine reading comprehension dataset of 120,000 question-answer pairs.

- Documents are CNN news articles.
- Questions are written by human users in natural language.
- Answers may be multiword passages of the source text.
- Questions may be unanswerable.
- NewsQA is collected using a 3-stage, siloed process.
- Questioners see only an article’s headline and highlights.
- Answerers see the question and the full article, then select an answer passage.
- Validators see the article, the question, and a set of answers that they rank.
- NewsQA is more natural and more challenging than previous datasets.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| NewsQA  | $97,850$  | $5,486$   | $5,396$  | $726.8$             | $5.0$               |

### Data Sample

Input

```
"What is going live on Tuesday ? [X_SEP] ( CNN ) -- Comcast rolled out a Web-based on-demand television and movie service on Tuesday that gives customers access to more than 2,000 hours of television and movies . The move comes as users increasingly are bypassing their TV sets and heading straight to the Web -- both legally and illegally -- to watch their favorite shows . The service , named Fancast XFINITY TV ( formerly called TV Everywhere ) , is the biggest cable industry initiative to keep people from skipping traditional TV service in the United States . `` I watch TV online every day . I find it more convenient than my regular TV ... , '' Michael Heard , a self-employed computer repairman from Atlanta , Georgia , said via e-mail . `` I 'm usually watching TV on one window while reading e-mail or tweets on another . `` And also my time is important , so sitting down and watching a show at 8 or 9 p.m. is n't convenient . Online TV allows me to watch what I want when I want . '' Networks have tried for the past couple of years to find a way to reach Web-watching audiences by streaming content on their Web sites or making partnerships with Hulu , one of the larger online TV sites . Now , Comcast is hoping it can make a dent in the market by serving up premium content . It is available to all Comcast customers , so long as they subscribe to both Internet and Cable service . Heard said he does n't expect to give up Hulu , though many of the same episodes and Web clips will be available on Fancast . Heard , trying out the site Tuesday after it went live , tweeted that he thought the service was `` awesome , '' and he finally had a place where he could watch the entire series of `` The Sopranos . '' `` The quality , it 's really clear and loads fast , '' he said , though he noted there were still some bugs in the product . Heard occasionally had to reload the site , and felt that installing the video player and authorizing the computer took a bit more time and was harder than simply pressing play on Hulu . The service is getting mixed reviews on Twitter , with customers giving instant feedback about their experiences . Some complained that high-definition videos , which are available on Hulu , are not available on Comcast 's service . Others complained about having to download a separate video player , the service not being compatible with the Linux operating system , and only being able to authorize the service on a total of three computers . Comcast hopes to wow customers even more in the future . In the next six months , after the service has gone through more beta testing , the company plans to open the service to a broader customer base . Customers will be able to access all content -- depending on the tiered level of service they subscribe to . Those not paying for HBO regularly , for example , wo n't be able to snag the newest episode of `` Curb Your Enthusiasm '' online . In addition to catching up with sites like Hulu and Clicker , Comcast executives told media outlets they expect to serve up a feature that Tivo fans have come to love -- allowing customers to program their DVR from afar . Executives said they hope the service would be available in about six months . To access the content , users simply need to log in with their Comcast e-mail address at fancast.com . The site is offering live online help , including help retrieving those addresses ."
```

Output

```
'Web-based on-demand television and movie service'
```

## LeaderBoard

Descending order by F1.

| Model                                                | F1     | EM     | Repository                                                | Generated Text |
| ---------------------------------------------------- | ------ | ------ | --------------------------------------------------------- | -------------- |
| [ SpanBERT](https://arxiv.org/pdf/1907.10529v3.pdf)  | $73.6$ |        | [Official](https://github.com/facebookresearch/SpanBERT)  |                |
| [ LinkBERT](https://arxiv.org/pdf/2203.15827v1.pdf)  | $72.6$ |        | [Official](https://github.com/michiyasunaga/LinkBERT)     |                |
| [ DecaProp](https://arxiv.org/pdf/1811.04210v2.pdf)  | $66.3$ | $53.1$ | [Official](https://github.com/vanzytay/NIPS2018_DECAPROP) |                |
| [ AMANDA](https://arxiv.org/pdf/1801.08290v1.pdf)    | $63.7$ | $48.4$ | [Official](https://github.com/nusnlp/amanda)              |                |
| [ MINIMAL](https://arxiv.org/pdf/1805.08092v1.pdf)   | $63.2$ | $50.1$ |                                                           |                |
| [ FastQAExt](https://arxiv.org/pdf/1703.04816v3.pdf) | $56.1$ | $43.7$ |                                                           |                |

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

