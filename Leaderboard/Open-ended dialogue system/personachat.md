# PersonaChat

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/P18-1205.pdf)

Repository: [Official]()

The PERSONA-CHAT dataset is a crowd-sourced dataset, collected via Amazon Mechanical Turk, where each of the pair of speakers condition their dialogue on a given profile, which is provided.

### Overview

| Dataset     | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ----------- | --------- | --------- | -------- | ------------------- | ------------------- |
| PersonaChat | $122,499$ | $14,602$  | $14,056$ | $132.1$             | $11.9$              |

### Data Sample

Input

```
'i read twenty books a year . [SEP] i am a stunt double as my second job . [SEP] i only eat kosher . [SEP] i was raised in a single parent household . [X_SEP] hello what are doing today ?'
```

Output

```
'i am good , i just got off work and tired , i have two jobs .'
```

## LeaderBoard

Descending order by Avg F1.

| Model                                                        | Avg F1  | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ | -------------- |
| [P$^2$ Bot](https://arxiv.org/pdf/2004.05388v1.pdf)          | $19.77$ | [Official](https://github.com/SivilTaram/Persona-Dialogue-Generation) |                |
| [TransferTransfo](https://arxiv.org/pdf/1901.08149v2.pdf)    | $19.09$ | [Official](https://github.com/huggingface/transfer-learning-conv-ai) |                |
| [Seq2Seq + Attention](https://arxiv.org/pdf/1409.0473v7.pdf) | $16.18$ |                                                              |                |
| [ KV Profile Memory](https://arxiv.org/pdf/1801.07243v5.pdf) | $11.9$  | [Official](https://github.com/facebookresearch/ParlAI)       |                |

## Citation

```
 @inproceedings{zhang-etal-2018-personalizing,
    title = "Personalizing Dialogue Agents: {I} have a dog, do you have pets too?",
    author = "Zhang, Saizheng  and
      Dinan, Emily  and
      Urbanek, Jack  and
      Szlam, Arthur  and
      Kiela, Douwe  and
      Weston, Jason",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-1205",
    doi = "10.18653/v1/P18-1205",
    pages = "2204--2213",
}
```