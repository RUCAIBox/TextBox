# WOZ

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/P17-1163.pdf)

The WoZ 2.0 dataset is a newer dialogue state tracking dataset whose evaluation is detached from the noisy output of speech recognition systems. Similar to DSTC2, it covers the restaurant search domain and has identical evaluation.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| WOZ     | $6,364$   | $1,260$   | -        | $47.0$              | $10.6$              |

### Data Sample

```
[ 
	{ 
		"turn_label": [ [ "food", "eritrean" ] ], 
		"asr": [ [ "Are there any eritrean restaurants in town?" ] ], 
		"system_transcript": "", 
		"turn_idx": 0, 
		"belief_state": [ { "slots": [ [ "food", "eritrean" ] ], "act": "inform" } ], 
		"transcript": "Are there any eritrean restaurants in town?", 
		"system_acts": [] 
	}, 
	{ 
		"turn_label": [ [ "food", "chinese" ] ], 
		"asr": [ [ "How about Chinese food?" ] ], 
		"system_transcript": "No, there are no eritrean restaurants in town. Would you like a different restaurant? ", 
		"turn_idx": 1, 
		"belief_state": [ { "slots": [ [ "food", "chinese" ] ], "act": "inform" } ], 
		"transcript": "How about Chinese food?", 
		"system_acts": [] 
	}, 
	...
]
```

## LeaderBoard

Descending order by Joint.

| Model                                                        | Joint   | Repository                                                   | Generated Text |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ | -------------- |
| [AG-DST](https://arxiv.org/pdf/2110.15659v1.pdf)             | $91.37$ | [Official](https://github.com/PaddlePaddle/Knover/tree/develop/projects/AG-DST) |                |
| [Seq2Seq-DU-w/oSchema](https://arxiv.org/pdf/2011.09553v2.pdf) | $91.2$  | [Official](https://github.com/sweetalyssum/Seq2Seq-DU)       |                |
| [T5(span)](https://arxiv.org/pdf/2108.13990v2.pdf)           | $91$    |                                                              |                |

## Citation

```
@inproceedings{woz,
    title = "Neural Belief Tracker: Data-Driven Dialogue State Tracking",
    author = "Mrk{\v{s}}i{\'c}, Nikola  and
      {\'O} S{\'e}aghdha, Diarmuid  and
      Wen, Tsung-Hsien  and
      Thomson, Blaise  and
      Young, Steve",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1163",
    doi = "10.18653/v1/P17-1163",
    pages = "1777--1788",
}
```

