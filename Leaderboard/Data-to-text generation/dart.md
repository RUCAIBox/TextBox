# DART

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/2007.02871)

Repository: [Official](https://github.com/Yale-LILY/dart)

The dataset is an open domain structured **DA**ta **R**ecord to **T**ext generation dataset with over 82k instances (DARTs).

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| DART    | $62,659$  | $6,980$   | $12,552$ | $27.8$              | $19.3$              |

### Data Sample

Input

```

```

Output

```

```

## LeaderBoard

Descending order by BLEU.

| Model                                                       | BLEU    | METEOR | Repository | Generated Text |
| ----------------------------------------------------------- | ------- | ------ | ---------- | -------------- |
| [T5](https://arxiv.org/abs/2007.02871)                      | $50.66$ | $0.40$ |            |                |
| [BART](https://arxiv.org/abs/2007.02871)                    | $48.56$ | $0.39$ |            |                |
| [LSTM with Attention](https://arxiv.org/abs/2007.02871)     | $29.66$ | $0.27$ |            |                |
| [End-to-End Transformers](https://arxiv.org/abs/2007.02871) | $27.24$ | $0.25$ |            |                |

## Citation

```
 @inproceedings{dart,
    title = "{DART}: Open-Domain Structured Data Record to Text Generation",
    author = "Nan, Linyong  and
      Radev, Dragomir  and
      Zhang, Rui  and
      Rau, Amrit  and
      Sivaprasad, Abhinand  and
      Hsieh, Chiachun  and
      Tang, Xiangru  and
      Vyas, Aadit  and
      Verma, Neha  and
      Krishna, Pranav  and
      Liu, Yangxiaokang  and
      Irwanto, Nadia  and
      Pan, Jessica  and
      Rahman, Faiaz  and
      Zaidi, Ahmad  and
      Mutuma, Mutethia  and
      Tarabar, Yasin  and
      Gupta, Ankit  and
      Yu, Tao  and
      Tan, Yi Chern  and
      Lin, Xi Victoria  and
      Xiong, Caiming  and
      Socher, Richard  and
      Rajani, Nazneen Fatema",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.37",
    doi = "10.18653/v1/2021.naacl-main.37",
    pages = "432--447",
}
```