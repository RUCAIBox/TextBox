# Topical-Chat

## Dataset

### Instruction

Paper: [Paper](https://www.isca-speech.org/archive/interspeech_2019/gopalakrishnan19_interspeech.html)

Repository: [Official](https://github.com/alexa/Topical-Chat)

A knowledge-grounded human-human conversation dataset where the underlying knowledge spans 8 broad topics and conversation partners don’t have explicitly defined roles.

### Overview

| Dataset      | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------------ | --------- | --------- | -------- | ------------------- | ------------------- |
| Topical-Chat | $179,750$ | $22,295$  | $22,452$ | $223.3$             | $20.0$              |

### Data Sample

```
{
<conversation_id>: {
	“article_url”: <article url>,
	“config”: <config>, # one of A,B,C, D
	“content”: [ # ordered list of conversation turns
		{ 
		“agent”: “agent_1”, # or “agent_2”,
		“message” : <message text>,
		“sentiment”: <text>,
		“knowledge_source” : [“AS1”, “Personal Knowledge”,...],
		“turn_rating”: “Poor”, # Note: changed from number to actual annotated text
		},…
	],
	“conversation_rating”: {
		“agent_1”: “Good”,
		“agent_2”: “Excellent”
		}
},…
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
 @inproceedings{tc,
  author    = {Karthik Gopalakrishnan and
               Behnam Hedayatnia and
               Qinglang Chen and
               Anna Gottardi and
               Sanjeev Kwatra and
               Anu Venkatesh and
               Raefer Gabriel and
               Dilek Hakkani{-}T{\"{u}}r},
  title     = {Topical-Chat: Towards Knowledge-Grounded Open-Domain Conversations},
  booktitle = {Interspeech 2019, 20th Annual Conference of the International Speech
               Communication Association},
  pages     = {1891--1895},
  publisher = {{ISCA}},
  year      = {2019},
  url={https://doi.org/10.21437/Interspeech.2019-3079},
  doi       = {10.21437/Interspeech.2019-3079},
}
```