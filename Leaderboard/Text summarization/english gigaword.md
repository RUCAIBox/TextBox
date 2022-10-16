# English GigaWord

## Dataset

### Instruction

Paper: [Paper](http://dx.doi.org/10.18653/v1/D15-1044)

Headline-generation on a corpus of article pairs from Gigaword consisting of around 4 million articles.

### Overview

| Dataset          | Num Train   | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ---------------- | ----------- | --------- | -------- | ------------------- | ------------------- |
| English GigaWord | $8,888,486$ | $987,610$ | -        | $366.9$             | $24.3$              |

### Data Sample

Input

```
australia 's current account deficit shrunk by a record #.## billion dollars -lrb- #.## billion us -rrb- in the june quarter due to soaring commodity prices , figures released monday showed .
```

Output

```
australian current account deficit narrows sharply
```

## LeaderBoard

Descending order by ROUGE-2.

| Model                                                       | ROUGE-1 | ROUGE-2 | ROUGE-L | Repository                                                   | Generated Text |
| ----------------------------------------------------------- | ------- | ------- | ------- | ------------------------------------------------------------ | -------------- |
| [BART-RXF](https://arxiv.org/pdf/2008.03156v1.pdf)          | $40.45$ | $20.69$ | $36.56$ | [Official](https://github.com/facebookresearch/fairseq/tree/main/examples/rxf) |                |
| [OFA](https://arxiv.org/pdf/2202.03052v2.pdf)               | $39.81$ | $20.66$ | $37.11$ | [Official](https://github.com/ofa-sys/ofa)                   |                |
| [MUPPET BART Large](https://arxiv.org/pdf/2101.11038v1.pdf) | $40.4$  | $20.54$ | $36.21$ | [Official](https://huggingface.co/facebook/muppet-roberta-large) |                |
| [ControlCopying](https://arxiv.org/pdf/1911.10390v1.pdf)    | $39.08$ | $20.47$ | $36.69$ | [Official](https://github.com/ucfnlp/control-over-copying)   |                |
| [Transformer+Wdrop](https://arxiv.org/pdf/2104.01853v1.pdf) | $39.66$ | $20.45$ | $36.59$ | [Official](https://github.com/takase/rethink_perturbations)  |                |
| [ProphetNet](https://arxiv.org/pdf/2001.04063v3.pdf)        | $39.51$ | $20.42$ | $36.69$ | [Official](https://github.com/microsoft/ProphetNet)          |                |
| [ PALM](https://arxiv.org/pdf/2004.07159v2.pdf)             | $39.45$ | $20.37$ | $36.75$ | [Official](https://github.com/alibaba/AliceMind/tree/main/PALM) |                |
| [UniLM](https://arxiv.org/pdf/1905.03197v3.pdf)             | $38.90$ | $20.05$ | $36.00$ | [Official](https://github.com/microsoft/unilm)               |                |
| [ PEGASUS](https://arxiv.org/pdf/1912.08777v2.pdf)          | $39.12$ | $19.86$ | $36.24$ | [Official](https://github.com/google-research/pegasus)       |                |
| [BiSET](https://arxiv.org/pdf/1906.05012v1.pdf)             | $39.11$ | $19.78$ | $36.87$ | [Official](https://github.com/InitialBug/BiSET)              |                |
| [MASS](https://arxiv.org/pdf/1905.02450v5.pdf)              | $38.73$ | $19.71$ | $35.96$ | [Official](https://github.com/microsoft/MASS)                |                |
| [Re$^3$Sum](https://aclanthology.org/P18-1015.pdf)          | $37.04$ | $19.03$ | $34.46$ |                                                              |                |
| [Transformer](https://arxiv.org/pdf/1706.03762v5.pdf)       | $37.57$ | $18.90$ | $34.69$ |                                                              |                |

## Citation

```
 @article{Rush_2015,
   title={A Neural Attention Model for Abstractive Sentence Summarization},
   url={http://dx.doi.org/10.18653/v1/D15-1044},
   DOI={10.18653/v1/d15-1044},
   journal={Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
   publisher={Association for Computational Linguistics},
   author={Rush, Alexander M. and Chopra, Sumit and Weston, Jason},
   year={2015}
}
```