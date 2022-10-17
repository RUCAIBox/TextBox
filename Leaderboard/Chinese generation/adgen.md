# ADGEN

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1908.06605)

Repository: [Official](https://github.com/ZhihongShao/Planning-based-Hierarchical-Variational-Model)

The dataset is constructed from a Chinese e-commerce platform. The dataset consists of 119K pairs of advertising text and clothing specification table. Each table is a set of attribute-value pairs describing a piece of clothing. Authors made some modifications to the original specification tables. Specifically, if some attribute-value pairs from a table do not occur in the corresponding text, the pairs are removed from the table. Authors also recognized attribute values by string matching with a dictionary of attribute values. If a pair occurs in the text but not in the table, the pair is added to the table.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| ADGEN   | $114,599$ | $1,070$   | $3,127$  | 78.3                | 113.4               |

### Data Sample

Input
```
[[类型, 裤], [风格, 简约], [风格, 潮], [图案, 格子], [图案, 几何], [图案, 线条], [裤长, 七分裤], [裤型, 阔腿裤]]
```
Output
```
这款阔腿裤，整体设计简约利落，时尚的阔腿款式带来鲜明的几何设计美感，褪去传统装束的厚重与臃肿，更具轻盈美感。 搭配七分裤长修饰出挺拔的腿部线条，气质的格纹图案不显单调，尽显女性优雅气质。 斜门襟设计潮流出众，让你时刻保持动人的女性风采。
```
## LeaderBoard

Descending order by BLEU-4.

| Model                                             | BLEU-4  | Repository                                                   | Generated Text |
| ------------------------------------------------- | ------- | ------------------------------------------------------------ | -------------- |
| [ERNIE 3.0](https://arxiv.org/pdf/2107.02137.pdf) | $30.16$ | [Official](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0) |                |
| [CPT](https://arxiv.org/abs/2109.05729)           | $10.7$  | [Official](https://github.com/fastnlp/CPT)                   |                |
| [CPM-2](https://arxiv.org/abs/2109.05729)         | $10.6$  |                                                              |                |
| [BART](https://arxiv.org/abs/2109.05729)          | $10.0$  |                                                              |                |
| [mT5](https://arxiv.org/pdf/2107.02137.pdf)       | $9.82$  |                                                              |                |
| [mBART](https://arxiv.org/abs/2109.05729)         | $8.5$   |                                                              |                |

## Citation

```@inproceedings{adgen,
@inproceedings{adgen,
    title = "Long and Diverse Text Generation with Planning-based Hierarchical Variational Model",
    author = "Shao, Zhihong  and
      Huang, Minlie  and
      Wen, Jiangtao  and
      Xu, Wenfei  and
      Zhu, Xiaoyan",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1321",
    doi = "10.18653/v1/D19-1321",
    pages = "3257--3268",
}
```