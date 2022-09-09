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