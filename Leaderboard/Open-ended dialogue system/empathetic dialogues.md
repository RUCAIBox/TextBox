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

```
{
    "context": "sentimental",
    "conv_id": "hit:0_conv:1",
    "prompt": "I remember going to the fireworks with my best friend. There was a lot of people_comma_ but it only felt like us in the world.",
    "selfeval": "5|5|5_2|2|5",
    "speaker_idx": 1,
    "tags": "",
    "utterance": "I remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people_comma_ we felt like the only people in the world.",
    "utterance_idx": 1
}
```

## LeaderBoard

Descending order by BLEU-4.

| Model                                                        | BLEU-4 | F1     | ROUGE-L | Repository | Generated Text |
| ------------------------------------------------------------ | ------ | ------ | ------- | ---------- | -------------- |
| [Multi-Modal BlenderBot](https://arxiv.org/pdf/2010.01082v1.pdf) | $1.5$  | $19.2$ | $24.5$  |            |                |

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