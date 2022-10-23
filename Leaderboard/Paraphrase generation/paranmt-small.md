# ParaNMT-small

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/P19-1599.pdf)

Repository: [Official](https://github.com/mingdachen/syntactic-template-generation)

 ParaNMT-small (Chen et al., 2019a) contains 500K sentence-paraphrase pairs for training, and 1300 manually labeled sentence-exemplar-reference which is further split into 800 test data points and 500 dev. data points respectively.

### Overview

| Dataset       | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------------- | --------- | --------- | -------- | ------------------- | ------------------- |
| ParaNMT-small | $493,081$ | $500$     | $800$    | $62.6$              | $29.4$              |

### Data Sample

Input

```
i had a dream yesterday and it was about you <sep>  (ROOT (S (S (VP (LS i) (S (VP (VBD had) (NP (DT a) (NN dream)) (NP (NN yesterday)))))) (CC and) (S (NP (PRP it)) (VP (VBD was) (PP (IN about) (NP (PRP you))))))) <sep> (ROOT (S (NP (NN )) (VP (VBD ) (NP (DT ) (NN )) (PP (IN ) (NP )))))
```

Output

```
i had a dream about you yesterday
```

## LeaderBoard

Descending order by BLEU.

| Model                                                     | BLEU   | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | ST    | Repository                                                   | Generated Text |
| --------------------------------------------------------- | ------ | ------- | ------- | ------- | ------ | ----- | ------------------------------------------------------------ | -------------- |
| [SCPN](https://aclanthology.org/P19-1599.pdf)             | $19.2$ | $50.4$  | $26.1$  | $53.5$  | $28.4$ | $5.9$ |                                                              |                |
| [Source-as-Output](https://aclanthology.org/P19-1599.pdf) | $18.5$ | $50.6$  | $23.2$  | $47.7$  | $12.1$ | $5.9$ |                                                              |                |
| [VGVAE](https://aclanthology.org/P19-1599.pdf)            | $13.6$ | $44.7$  | $21.0$  | $48.3$  | $24.8$ | $6.7$ | [Official](https://github.com/mingdachen/syntactic-template-generation) |                |


## Citation

```
@inproceedings{chen-etal-2019-controllable,
    title = "Controllable Paraphrase Generation with a Syntactic Exemplar",
    author = "Chen, Mingda  and
      Tang, Qingming  and
      Wiseman, Sam  and
      Gimpel, Kevin",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1599",
    doi = "10.18653/v1/P19-1599",
    pages = "5972--5984",
}
```