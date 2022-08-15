# WikiTableT

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/2021.findings-acl.17.pdf)

Repository: [Official](https://github.com/mingdachen/WikiTableT)

Authors cast generating Wikipedia sections as a data-to-text generation task and create a large-scale dataset, WikiTableT, that pairs Wikipedia sections with their corresponding tabular data and various metadata. WikiTableT contains millions of instances, covering a broad range of topics, as well as a variety of flavors of generation tasks with different levels of flexibility.

### Overview

| Dataset    | Num Train   | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ---------- | ----------- | --------- | -------- | ------------------- | ------------------- |
| WikiTableT | $1,453,794$ | $4,533$   | $4,351$  | $81.0$              | $99.7$              |

### Data Sample

Input

```

```

Output

```

```

## LeaderBoard

Descending order by METRIC.

| Model                                                        | BLEU   | ROUGE-L | METEOR | Repository | Generated Text |
| ------------------------------------------------------------ | ------ | ------- | ------ | ---------- | -------------- |
| [transformer+beam search](https://aclanthology.org/2021.findings-acl.17.pdf) | $19.5$ | $39.9$  | $23.9$ |            |                |
| [transformer+beam search+$n$-gram blocking](https://aclanthology.org/2021.findings-acl.17.pdf) | $19.3$ | $39.3$  | $24.4$ |            |                |
| [transformer+greedy](https://aclanthology.org/2021.findings-acl.17.pdf) | $18.9$ | $38.5$  | $23.5$ |            |                |
| [transformer+nucleus sampling](https://aclanthology.org/2021.findings-acl.17.pdf) | $18.3$ | $36.1$  | $23.7$ |            |                |

## Citation

```
 @inproceedings{chen-etal-2021-wikitablet,
    title = "{W}iki{T}able{T}: A Large-Scale Data-to-Text Dataset for Generating {W}ikipedia Article Sections",
    author = "Chen, Mingda  and
      Wiseman, Sam  and
      Gimpel, Kevin",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.17",
    doi = "10.18653/v1/2021.findings-acl.17",
    pages = "193--209",
}
```