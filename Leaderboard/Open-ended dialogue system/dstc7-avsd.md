# DSTC7-AVSD

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/pdf/1806.00525v1.pdf)

Homepage: [Official](http://workshop.colips.org/dstc7/call.html)

Repository: [Official](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge)

The Audio Visual Scene-Aware Dialog (AVSD) dataset, or DSTC7 Track 3, is a audio-visual dataset for dialogue understanding. The goal with the dataset and track was to design systems to generate responses in a dialog about a video, given the dialog history and audio-visual content of the video.

### Overview

| Dataset    | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ---------- | --------- | --------- | -------- | ------------------- | ------------------- |
| DSTC7-AVSD | $76,590$  | $17,870$  | $1,710$  | $148.2$             | $11.5$              |

### Data Sample

Input

```

```

Output

```

```

## LeaderBoard

Descending order by CIDEr.

| Model                                                        | CIDEr   | Repository                                        | Generated Text |
| ------------------------------------------------------------ | ------- | ------------------------------------------------- | -------------- |
| [simple](https://openaccess.thecvf.com/content_CVPR_2019/papers/Schwartz_A_Simple_Baseline_for_Audio-Visual_Scene-Aware_Dialog_CVPR_2019_paper.pdf) | $0.941$ | [Official](https://github.com/idansc/simple-avsd) |                |

## Citation

```
 @article{da,
  title={Audio visual scene-aware dialog (avsd) challenge at dstc7},
  author={Alamri, Huda and Cartillier, Vincent and Lopes, Raphael Gontijo and Das, Abhishek and Wang, Jue and Essa, Irfan and Batra, Dhruv and Parikh, Devi and Cherian, Anoop and Marks, Tim K and others},
  journal={arXiv preprint arXiv:1806.00525},
  url={http://arxiv.org/abs/1806.00525},
  year={2018}
}
```