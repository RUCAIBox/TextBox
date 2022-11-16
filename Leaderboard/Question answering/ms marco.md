# MS MARCO

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1611.09268)

Homepage: [Official](https://microsoft.github.io/msmarco/)

The MS MARCO (Microsoft MAchine Reading Comprehension) is a collection of datasets focused on deep learning in search. The first dataset was a question answering dataset featuring 100,000 real Bing questions and a human generated answer. Over time the collection was extended with a 1,000,000 question dataset, a natural language generation dataset, a passage ranking dataset, keyphrase extraction dataset, crawling dataset, and a conversational search.

### Overview

| Dataset  | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | -------- | ------------------- | ------------------- |
| MS MARCO | $681,445$ | $77,580$  | -        | $68.7$              | $13.3$              |

### Data Sample

Input

```
why did rachel carson write an obligation to endure [X_SEP] The Obligation to Endure by Rachel Carson Rachel Carson's essay on The Obligation to Endure, is a very convincing argument about the harmful uses of chemicals, pesticides, herbicides, and fertilizers on the environment. [SEP] Carson believes that as man tries to eliminate unwanted insects and weeds, however he is actually causing more problems by polluting the environment with, for example, DDT and harming living things. Carson adds that the intensification of agriculture is causing other major problems, like newly developed or created insects and diseases."
```

Output

```
Rachel Carson writes The Obligation to Endure because believes that as man tries to eliminate unwanted insects and weeds, however he is actually causing more problems by polluting the environment.'
```

## LeaderBoard

Descending order by ROUGE-L.

| Model                                                       | ROUGE-L | BLEU-1  | Repository                                         | Generated Text |
| ----------------------------------------------------------- | ------- | ------- | -------------------------------------------------- | -------------- |
| [Multi-doc Enriched BERT]()                                 | $54.0$  | $56.5$  |                                                    |                |
| [Human]()                                                   | $53.9$  | $48.5$  |                                                    |                |
| [BERT Encoded T-Net]()                                      | $52.6$  | $53.9$  |                                                    |                |
| [ Masque Q&A Style](https://arxiv.org/pdf/1901.02262v2.pdf) | $52.2$  | $43.77$ |                                                    |                |
| [ Deep Cascade QA](https://arxiv.org/pdf/1811.11374v1.pdf)  | $52.01$ | $54.64$ |                                                    |                |
| [VNET](https://arxiv.org/pdf/1805.02220v2.pdf)              | $51.63$ | $54.37$ |                                                    |                |
| [BiDaF Baseline](https://arxiv.org/pdf/1611.01603v6.pdf)    | $23.96$ | $10.64$ | [Official](https://github.com/allenai/bi-att-flow) |                |

## Citation

```
@inproceedings{marco,
  author    = {Tri Nguyen and
               Mir Rosenberg and
               Xia Song and
               Jianfeng Gao and
               Saurabh Tiwary and
               Rangan Majumder and
               Li Deng},
  title     = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  booktitle = {CoCo@NIPS},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {1773},
  publisher = {CEUR-WS.org},
  url={http://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper9.pdf},
  year      = {2016}
}
```

