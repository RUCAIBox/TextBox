# MuTual

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/2020.acl-main.130.pdf)

Repository: [Official](https://github.com/Nealcly/MuTual)

MuTual is a retrieval-based dataset for multi-turn dialogue reasoning, which is modified from Chinese high school English listening comprehension test data. It tests dialogue reasoning via next utterance prediction.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| MuTual  | $33,691$  | $4,090$   | $3,248$  | $53.6$              | $14.5$              |

### Data Sample

```
{
	"answers": " ", 
	"options": [
		"m : although you 've got a flu , you did n't feel sick last night . you 've got better .", 
		"m : then the fact that you felt sick last night may have something to do with what you ate .", 
		"m : although you have sleep problems , you slept very well last night , did n't you ?", 
		"m : your nausea last night must have been caused by the flu , for you have never eaten in a restaurant ."
	], 
	"article": "m : you look rather pale . are you	 feeling well ? f : not very . i was sick most of the night . i did n't sleep very well . m : what seems to be the matter ? is it the flu ? f : no , i think it was something i ate . we ate at that new restaurant last night and i must have eaten something that did n't agree with me .", 
	"id": "test_1"
}	
```

## LeaderBoard

Descending order by METRIC.

| Model | Metric | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
|       |        |            |                |
|       |        |            |                |
|       |        |            |                |

## Citation

```
 @inproceedings{cui-etal-2020-mutual,
    title = "{M}u{T}ual: A Dataset for Multi-Turn Dialogue Reasoning",
    author = "Cui, Leyang  and
      Wu, Yu  and
      Liu, Shujie  and
      Zhang, Yue  and
      Zhou, Ming",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.130",
    doi = "10.18653/v1/2020.acl-main.130",
    pages = "1406--1416",
}
```