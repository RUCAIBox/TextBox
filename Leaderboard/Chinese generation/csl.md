# CSL

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/pdf/2209.05034v1.pdf)

Repository: [Official](https://github.com/CLUEbenchmark/CLGE)

Scientific literature serves as a high-quality corpus, supporting a lot of Natural Language Processing (NLP) research. However, existing datasets are centered around the English language, which restricts the development of Chinese scientific NLP. In this work, we present CSL, a large-scale Chinese Scientific Literature dataset, which contains the titles, abstracts, keywords and academic fields of 396k papers. To our knowledge, CSL is the first scientific document dataset in Chinese. The CSL can serve as a Chinese corpus. Also, this semi-structured data is a natural annotation that can constitute many supervised NLP tasks. Based on CSL, we present a benchmark to evaluate the performance of models across scientific domain tasks, i.e., summarization, keyword generation and text classification. We analyze the behavior of existing text-to-text models on the evaluation tasks and reveal the challenges for Chinese scientific NLP tasks, which provides a valuable reference for future research.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| CSL     | $3,000$   | $500$     | -        | 202.4               | 18.8                |

### Data Sample

Input
```
抽象了一种基于中心的战术应用场景与业务,并将网络编码技术应用于此类场景的实时数据多播业务中。在分析基于中心网络与Many-to-all业务模式特性的基础上,提出了仅在中心节点进行编码操作的传输策略以及相应的贪心算法。分析了网络编码多播策略的理论增益上界,仿真试验表明该贪心算法能够获得与理论相近的性能增益。最后的分析与仿真试验表明,在这种有中心网络的实时数据多播应用中,所提出的多播策略的实时性能要明显优于传统传输策略。
```
Output
```
网络编码在实时战术数据多播中的应用
```
## LeaderBoard

Descending order by ROUGE-L.

| Model                                         | ROUGE-L | Repository                                 | Generated Text |
| --------------------------------------------- | ------- | ------------------------------------------ | -------------- |
| [BART](https://arxiv.org/pdf/2109.05729.pdf)  | $64.2$  |                                            |                |
| [CPT](https://arxiv.org/pdf/2109.05729.pdf)   | $63.7$  | [Official](https://github.com/fastnlp/CPT) |                |
| [mT5](https://arxiv.org/pdf/2109.05729.pdf)   | $61.8$  |                                            |                |
| [mBART](https://arxiv.org/pdf/2109.05729.pdf) | $55.2$  |                                            |                |

## Citation

```
@misc{csl,
  title        = "{Chinese Language Generation Evaluation}",
  howpublished = "\url{https://github.com/CLUEbenchmark/CLGE}",
  note         = "Accessed: 2022-8-1"
}
```