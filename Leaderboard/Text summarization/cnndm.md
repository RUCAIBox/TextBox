# CNN/DailyMail

## Dataset

### Instruction

Paper: [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)

Repository: [Official](https://github.com/abisee/cnn-dailymail)

CNN/Daily Mail is a dataset for text summarization. Human generated abstractive summary bullets were generated from news stories in CNN and Daily Mail websites as questions (with one of the entities hidden), and stories as the corresponding passages from which the system is expected to answer the fill-in the-blank question. The authors released the scripts that crawl, extract and generate pairs of passages and questions from these websites.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| CNN/DM  | $287,227$ | $13,368$  | $11,490$ | $679.8$             | $48.3$              |

### Data Sample

Input
```
( CNN ) For the first time in eight years , a TV legend returned to doing what he does best . Contestants told to " come on down ! " on the April 1 edition of " The Price Is Right " encountered not host Drew Carey but another familiar face in charge of the proceedings . Instead , there was Bob Barker , who hosted the TV game show for 35 years before stepping down in 2007 . Looking spry at 91 , Barker handled the first price-guessing game of the show , the classic " Lucky Seven , " before turning hosting duties over to Carey , who finished up . Despite being away from the show for most of the past eight years , Barker did n't seem to miss a beat .
```
Output
```
Bob Barker returned to host " The Price Is Right " on Wednesday . [X_SEP] Barker , 91 , had retired as host in 2007 .
```
## LeaderBoard

Descending order by ROUGE-2.

| Model                                                        | ROUGE-1 | ROUGE-2 | ROUGE-L | Repository                                                   | Generated Text                                               |
| ------------------------------------------------------------ | ------- | ------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [BRIO](https://arxiv.org/abs/2203.16804)                     | $47.78$ | $23.55$ | $44.57$ | [Official](https://github.com/yixinL7/BRIO)                  |                                                              |
| [SummaReranker](https://arxiv.org/abs/2203.06569)            | $47.04$ | $22.32$ | $43.72$ | [Official](https://github.com/ntunlp/SummaReranker)          |                                                              |
| [GSum](https://arxiv.org/abs/2010.08014)                     | $45.94$ | $22.32$ | $42.48$ | [Official](https://github.com/neulab/guided_summarization)   | [Corpus](https://drive.google.com/drive/folders/1lfGRNkP0dxb9oRlIi0hO4zvYG_hUtcX8?usp=sharing) |
| [SimCLS](https://arxiv.org/abs/2106.01890)                   | $46.67$ | $22.15$ | $43.54$ | [Official](https://github.com/yixinL7/SimCLS)                |                                                              |
| [UniLM-v2](https://arxiv.org/abs/2002.12804)                 | $44.69$ | $21.89$ | $41.81$ | [Official](https://github.com/microsoft/unilm)               | [Corpus](https://drive.google.com/drive/folders/1E_yBKCWH9G3BpPRi7xdiEps9Em26zdAa?usp=sharing) |
| [SeqCo](https://arxiv.org/abs/2109.03481)                    | $45.02$ | $21.80$ | $41.75$ | [Official](https://github.com/xssstory/SeqCo)                |                                                              |
| [T5](https://arxiv.org/abs/1910.10683)                       | $43.52$ | $21.55$ | $40.69$ | [Official](https://github.com/google-research/text-to-text-transfer-transformer) | [Corpus](https://drive.google.com/drive/folders/1WfHPgx6o4jGF3riwYTUsejmScZjo888p?usp=sharing) |
| [PEGASUS](https://arxiv.org/abs/1912.08777)                  | $44.17$ | $21.47$ | $41.11$ | [Official](https://github.com/google-research/pegasus)       |                                                              |
| [BART](https://arxiv.org/abs/1910.13461)                     | $44.16$ | $21.28$ | $40.90$ | [Official](https://github.com/facebookresearch/fairseq/tree/main/examples/bart) | [Corpus](https://drive.google.com/drive/folders/1k1eROTpSe9cvoKLT1v3wXd_BRKME9ROn?usp=sharing) |
| [ProphetNet](https://arxiv.org/abs/2001.04063)               | $44.20$ | $21.17$ | $41.30$ | [Official](https://github.com/microsoft/ProphetNet)          |                                                              |
| [MatchSum](https://arxiv.org/abs/2004.08795)                 | $44.41$ | $20.86$ | $40.55$ | [Official](https://github.com/maszhongming/MatchSum)         |                                                              |
| [DiscoBERT](https://www.cs.utexas.edu/~jcxu/material/ACL20/DiscoBERT_ACL2020.pdf) | $43.77$ | $20.85$ | $40.67$ | [Official](https://github.com/jiacheng-xu/DiscoBERT)         |                                                              |
| [Refactor](https://arxiv.org/abs/2104.07210)                 | $44.13$ | $20.51$ | $40.29$ | [Official](https://github.com/yixinL7/Refactoring-Summarization) | [Corpus](https://drive.google.com/drive/folders/1Qgkphp1UEjlLfMPZ85h7LwpeVwfLBN0-?usp=sharing) |
| [BertSumExt](https://arxiv.org/abs/1908.08345)               | $43.85$ | $20.34$ | $39.90$ | [Official](https://github.com/nlpyang/PreSumm)               |                                                              |
| [UniLM-v1](https://arxiv.org/abs/1905.03197)                 | $43.33$ | $20.21$ | $40.51$ | [Official](https://github.com/microsoft/unilm)               | [Corpus](https://drive.google.com/drive/folders/1x7q0kl8BwpeS3Fa-uFapY_CfJDZhelTT?usp=sharing) |
| [GAN](https://arxiv.org/abs/1711.09357)                      | $39.92$ | $17.65$ | $36.71$ |                                                              |                                                              |

## Citation
```
@inproceedings{cnndm, 
   title = "Get To The Point: Summarization with Pointer-Generator Networks", 
    author = "See, Abigail  and 
      Liu, Peter J.  and 
      Manning, Christopher D.", 
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long apers)", 
    month = jul, 
    year = "2017", 
    address = "Vancouver, Canada", 
    publisher = "Association for Computational Linguistics", 
    url = "https://aclanthology.org/P17-1099", 
    doi = "10.18653/v1/P17-1099", 
    pages = "1073--1083", 
}
```
