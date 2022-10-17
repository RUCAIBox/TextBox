# SAMSum

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/D19-5409.pdf)

The SAMSum dataset contains about 16k messenger-like conversations with summaries. Conversations were created and written down by linguists fluent in English. Linguists were asked to create conversations similar to those they write on a daily basis, reflecting the proportion of topics of their real-life messenger conversations. The style and register are diversified - conversations could be informal, semi-formal or formal, they may contain slang words, emoticons and typos. Then, the conversations were annotated with summaries. It was assumed that summaries should be a concise brief of what people talked about in the conversation in third person. The SAMSum dataset was prepared by Samsung R&D Institute Poland and is distributed for research purposes (non-commercial licence: CC BY-NC-ND 4.0).

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| SAMSum  | $14,732$  | $818$     | $819$    | $103.4$             | $20.3$              |

### Data Sample

Input

```
Olivia: Who are you voting for in this election? Oliver: Liberals as always. Olivia: Me too!! Oliver: Great
```

Output

```
Olivia and Olivier are voting for liberals in this election.
```

## LeaderBoard

Descending order by ROUGE-2.

| Model                                               | ROUGE-1  | ROUGE-2  | ROUGE-L  | Repository                                                   | Generated Text |
| --------------------------------------------------- | -------- | -------- | -------- | ------------------------------------------------------------ | -------------- |
| [ConDigSum](https://arxiv.org/pdf/2109.04994v1.pdf) | $54.3$   | $29.3$   | $45.2$   | [Official](https://github.com/junpliu/condigsum)             |                |
| [HAT-CNNDM](https://arxiv.org/pdf/2104.07545v2.pdf) | $53.01$  | $28.27$  | -        |                                                              |                |
| [BART](https://arxiv.org/pdf/2109.04994v1.pdf)      | $52.6$   | $27$     | $42.1$   | [Official](https://github.com/junpliu/condigsum)             |                |
| [ssr-base]()                                        | $46.253$ | $21.337$ | $36.194$ | [Official](https://huggingface.co/santiviquez/ssr-base-finetuned-samsum-en) |                |
| [T5-small]()                                        | $40.039$ | $15.85$  | $31.808$ | [Official](https://huggingface.co/santiviquez/t5-small-finetuned-samsum-en) |                |
| [PTGen](https://arxiv.org/pdf/2109.04994v1.pdf)     | $40.1$   | $15.3$   | $36.6$   |                                                              |                |
| [LEAD3](https://arxiv.org/pdf/2109.04994v1.pdf)     | $31.4$   | $8.7$    | $29.4$   |                                                              |                |

## Citation

```
@inproceedings{gliwa-etal-2019-samsum,
    title = "{SAMS}um Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization",
    author = "Gliwa, Bogdan  and
      Mochol, Iwona  and
      Biesek, Maciej  and
      Wawer, Aleksander",
    booktitle = "Proceedings of the 2nd Workshop on New Frontiers in Summarization",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-5409",
    doi = "10.18653/v1/D19-5409",
    pages = "70--79",
}
```