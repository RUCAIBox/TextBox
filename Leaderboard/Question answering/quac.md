# QuAC

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/D18-1241.pdf)

Homepage: [Official](https://quac.ai/)

Repository: [Official](https://github.com/deepnlp-cs599-usc/quac)

**Qu**estion **A**nswering in **C**ontext is a dataset for modeling, understanding, and participating in information seeking dialog. Data instances consist of an interactive dialog between two crowd workers: (1) a *student* who poses a sequence of freeform questions to learn as much as possible about a hidden Wikipedia text, and (2) a *teacher* who answers the questions by providing short excerpts (spans) from the text. QuAC introduces challenges not found in existing machine comprehension datasets: its questions are often more open-ended, unanswerable, or only meaningful within the dialog context.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| QuAC    | $83,568$  | $31,906$  | -        | $487.9$             | $12.5$              |

### Data Sample

Input

```
'what happened in 1983? [X_SEP] In May 1983, she married Nikos Karvelas, a composer, with whom she collaborated in 1975 and in November she gave birth to her daughter Sofia. After their marriage, she started a close collaboration with Karvelas. Since 1975, all her releases have become gold or platinum and have included songs by Karvelas. In 1986, she participated at the Cypriot National Final for Eurovision Song Contest with the song Thelo Na Gino Star ("I Want To Be A Star"), taking second place. This song is still unreleased up to date. In 1984, Vissi left her record company EMI Greece and signed with CBS Records Greece, which later became Sony Music Greece, a collaboration that lasted until 2013. In March 1984, she released Na \'Hes Kardia ("If You Had a Heart"). The album was certified gold. The following year her seventh album Kati Simveni ("Something Is Happening") was released which included one of her most famous songs, titled "Dodeka" ["Twelve (O\'Clock)"] and reached gold status selling 80.000 units. In 1986 I Epomeni Kinisi ("The Next Move") was released. The album included the hit Pragmata ("Things") and went platinum, becoming the best selling record of the year. In February 1988 she released her ninth album Tora ("Now") and in December the album Empnefsi! ("Inspiration!") which went gold. In 1988, she made her debut as a radio producer on ANT1 Radio. Her radio program was titled after one of her songs Ta Koritsia Einai Atakta ("Girls Are Naughty") and was aired every weekend. In the same year, she participated with the song Klaio ("I\'m Crying") at the Greek National Final for Eurovision Song Contest, finishing third. In 1989, she released the highly successful studio album Fotia (Fire), being one of the first albums to feature western sounds. The lead single Pseftika ("Fake") became a big hit and the album reached platinum status, selling 180.000 copies and becoming the second best selling record of 1990. She performed at "Diogenis Palace" in that same year, Athens\'s biggest nightclub/music hall at the time. CANNOTANSWER'
```

Output

```
'In May 1983, she married Nikos Karvelas,'
```

## LeaderBoard

Descending order by F1.

| Model                                                        | F1     | Repository                                            | Generated Text |
| ------------------------------------------------------------ | ------ | ----------------------------------------------------- | -------------- |
| [Human]()                                                    | $81.1$ |                                                       |                |
| [RoR](https://arxiv.org/pdf/2109.04780.pdf)                  | $74.9$ | [Official](https://github.com/JD-AI-Research-NLP/RoR) |                |
| [MarCQAp](https://arxiv.org/pdf/2206.14796.pdf)              | $74.0$ | [Official](https://github.com/zorikg/MarCQAp)         |                |
| [GHR_ELECTRA](https://aclanthology.org/2022.findings-naacl.159.pdf) | $73.7$ | [Official](https://github.com/jaytsien/GHR)           |                |
| [Bert-FlowDelta](https://arxiv.org/pdf/1908.05117)           | $65.5$ | [Official](https://github.com/MiuLab/FlowDelta)       |                |
| [HAM](https://arxiv.org/pdf/1908.09456.pdf)                  | $65.4$ |                                                       |                |
| [GraphFlow](https://arxiv.org/pdf/1908.00059.pdf)            | $64.9$ |                                                       |                |
| [ FlowQA](https://arxiv.org/pdf/1810.06683v3.pdf)            | $64.1$ | [Official](https://github.com/hsinyuan-huang/FlowQA)  |                |
| [ GPT-3 175B](https://arxiv.org/pdf/2005.14165v4.pdf)        | $44.3$ | [Official](https://github.com/openai/gpt-3)           |                |

## Citation

```
@inproceedings{choi-etal-2018-quac,
    title = "{Q}u{AC}: Question Answering in Context",
    author = "Choi, Eunsol  and
      He, He  and
      Iyyer, Mohit  and
      Yatskar, Mark  and
      Yih, Wen-tau  and
      Choi, Yejin  and
      Liang, Percy  and
      Zettlemoyer, Luke",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1241",
    doi = "10.18653/v1/D18-1241",
    pages = "2174--2184",
}
```

