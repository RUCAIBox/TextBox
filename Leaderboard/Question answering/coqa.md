# CoQA

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1808.07042)

Homepage: [Official](https://stanfordnlp.github.io/coqa/)

CoQA is a large-scale dataset for building Conversational Question Answering systems. The goal of the CoQA challenge is to measure the ability of machines to understand a text passage and answer a series of interconnected questions that appear in a conversation.

CoQA contains 127,000+ questions with answers collected from 8000+ conversations. Each conversation is collected by pairing two crowdworkers to chat about a passage in the form of questions and answers. The unique features of CoQA include 1) the questions are conversational; 2) the answers can be free-form text; 3) each answer also comes with an evidence subsequence highlighted in the passage; and 4) the passages are collected from seven diverse domains. CoQA has a lot of challenging phenomena not present in existing reading comprehension datasets, e.g., coreference and pragmatic reasoning.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| CoQA    | $107,286$ | $31,621$  | -        | $349.4$             | $2.6$               |

### Data Sample

Input

```
what color was cotton ? [X_SEP] once upon a time , in a barn near a farm house , there lived a little white kitten named cotton . cotton lived high up in a nice warm place above the barn where all of the farmer ' s horses slept . but cotton wasn ' t alone in her little home above the barn , oh no . she shared her hay bed with her mommy and 5 other sisters . all of her sisters were cute and fluffy , like cotton . but she was the only white one in the bunch . the rest of her sisters were all orange with beautiful white tiger stripes like cotton ' s mommy . being different made cotton quite sad . she often wished she looked like the rest of her family . so one day , when cotton found a can of the old farmer ' s orange paint , she used it to paint herself like them . when her mommy and sisters found her they started laughing . " what are you doing , cotton ? ! " " i only wanted to be more like you " . cotton ' s mommy rubbed her face on cotton ' s and said " oh cotton , but your fur is so pretty and special , like you . we would never want you to be any other way " . and with that , cotton ' s mommy picked her up and dropped her into a big bucket of water . when cotton came out she was herself again . her sisters licked her face until cotton ' s fur was all all dry . " don ' t ever do that again , cotton ! " they all cried . " next time you might mess up that pretty white fur of yours and we wouldn ' t want that ! " then cotton thought , " i change my mind . i like being special " .
```

Output

```
white
```

## LeaderBoard

Descending order by Overall F1.

| Model                                                        | In-domain F1 | Out-of-domain F1 | Overall F1 | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | ------------ | ---------------- | ---------- | ------------------------------------------------------------ | -------------- |
| [RoBERTa + AT + KD](https://arxiv.org/pdf/1909.10772.pdf)    | $90.9$       | $89.2$           | $90.4$     |                                                              |                |
| [XLNet + Augmentation](https://github.com/stevezheng23/xlnet_extension_tf) | $89.9$       | $86.9$           | $89.0$     | [Official](https://github.com/stevezheng23/xlnet_extension_tf) |                |
| [Human](https://arxiv.org/abs/1808.07042)                    | $89.4$       | $87.4$           | $88.8$     |                                                              |                |
| [ GPT-3 175B](https://arxiv.org/pdf/2005.14165v4.pdf)        | -            | -                | $85$       | [Official](https://github.com/openai/gpt-3)                  |                |
| [BERT-large](https://arxiv.org/pdf/1810.04805v2.pdf)         | $82.5$       | $77.6$           | $81.1$     | [Official](https://github.com/google-research/bert)          |                |
| [SDNet](https://arxiv.org/pdf/1812.03593v5.pdf)              | $78.0$       | $73.1$           | $76.6$     | [Official](https://github.com/Microsoft/SDNet)               |                |
| [FlowQA](https://arxiv.org/pdf/1810.06683v3.pdf)             | $76.3$       | $71.8$           | $75.0$     | [Official](https://github.com/hsinyuan-huang/FlowQA)         |                |
| [BiDAF++](https://arxiv.org/pdf/1809.10735v2.pdf)            | $69.4$       | $63.8$           | $67.8$     | [Official](https://github.com/my89/co-squac)                 |                |
| [Augmt. DrQA](https://arxiv.org/abs/1808.07042)              | $67.6$       | $60.2$           | $65.4$     |                                                              |                |
| [DrQA+PGNet](https://arxiv.org/abs/1808.07042)               | $67.0$       | $60.4$           | $65.1$     |                                                              |                |
| [DrQA](https://arxiv.org/abs/1808.07042)                     | $54.5$       | $47.9$           | $52.6$     |                                                              |                |
| [PGNet](https://arxiv.org/abs/1808.07042)                    | $46.4$       | $38.3$           | $44.1$     |                                                              |                |
| [Seq2seq](https://arxiv.org/abs/1808.07042)                  | $27.7$       | $23.0$           | $26.3$     |                                                              |                |

## Citation

```
@article{coqa,
    title = "{C}o{QA}: A Conversational Question Answering Challenge",
    author = "Reddy, Siva  and
      Chen, Danqi  and
      Manning, Christopher D.",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "7",
    year = "2019",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q19-1016",
    doi = "10.1162/tacl_a_00266",
    pages = "249--266",
}
```

