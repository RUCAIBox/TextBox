# Empathetic Dialogues

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/P19-1534.pdf)

Repository: [Official](https://github.com/facebookresearch/EmpatheticDialogues)

The Empathetic Dialogues dataset is a large-scale multi-turn empathetic dialogue dataset collected on the Amazon Mechanical Turk, containing 24,850 one-to-one open-domain conversations. Each conversation was obtained by pairing two crowd-workers: a speaker and a listener. The speaker is asked to talk about the personal emotional feelings. The listener infers the underlying emotion through what the speaker says and responds empathetically. The dataset provides 32 evenly distributed emotion labels.

### Overview

| Dataset              | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------------------- | --------- | --------- | -------- | ------------------- | ------------------- |
| Empathetic Dialogues | $64,636$  | $9,308$   | $8,426$  | $52.7$              | $12.9$              |

### Data Sample

Input

```
'Today,as i was leaving for work in the morning,i had a tire burst in the middle of a busy road. That scared the hell out of me! [X_SEP] Today,as i was leaving for work in the morning,i had a tire burst in the middle of a busy road. That scared the hell out of me!'
```

Output

```
'Are you fine now?'
```

## LeaderBoard

Descending order by BLEU.

| Model                                                        | BLEU    | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ | -------------- |
| [Emotion-aware transformer encoder](https://arxiv.org/pdf/2204.11320v1.pdf) | $22.5$  | [Official](https://github.com/r9729104234/Emotion-Aware-Transformer-Encoder-for-Empathetic-Dialogue-Generation) |                |
| [CARO](https://arxiv.org/pdf/2204.11320v1.pdf)               | $17.9$  |                                                              |                |
| [Transformer](https://arxiv.org/pdf/2204.11320v1.pdf)        | $17.3$  |                                                              |                |
| [EmoDS Seq2Seq](https://arxiv.org/pdf/2204.11320v1.pdf)      | $17.3$  |                                                              |                |
| [EmpTransfo](https://arxiv.org/pdf/2204.11320v1.pdf)         | $15.92$ |                                                              |                |
| [Seq2Seq with Attention](https://arxiv.org/pdf/2204.11320v1.pdf) | $13.7$  |                                                              |                |

## Citation

```
 @inproceedings{rashkin-etal-2019-towards,
    title = "Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset",
    author = "Rashkin, Hannah  and
      Smith, Eric Michael  and
      Li, Margaret  and
      Boureau, Y-Lan",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1534",
    doi = "10.18653/v1/P19-1534",
    pages = "5370--5381",
}
```