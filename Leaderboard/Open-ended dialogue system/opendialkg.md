# OpenDialKG

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/P19-1081.pdf)

Repository: [Official](https://github.com/facebookresearch/opendialkg)

OpenDialKG contains utterance from 15K human-to-human role-playing dialogs is manually annotated with ground-truth reference to corresponding entities and paths from a large-scale KG with 1M+ facts.

### Overview

| Dataset    | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ---------- | --------- | --------- | -------- | ------------------- | ------------------- |
| OpenDialKG | $69,680$  | $7,743$   | -        | $54.2$              | $12.4$              |

### Data Sample

```
[
	{
		""message"": ""Do you like Iron Man"", 
		""type"": ""chat"", 
		""sender"": ""user""
	}, 
	{
		""metadata"": {""path"": [0.0, [[""Iron Man"", ""starred_actors"", ""Robert Downey Jr.""]], ""Iron Man is starring Robert Downey Jr.""]}, 
		""type"": ""action"", 
		""sender"": ""assistant"", 
		""action_id"": ""kgwalk/choose_path""
	}, 
	{
		""message"": ""Sure do! Robert Downey Jr. is a favorite."", 
		""type"": ""chat"", 
		""sender"": ""assistant""
	}, 
	{
		""metadata"": {""path"": [-0.5, [[""Robert Downey Jr."", ""~starred_actors"", ""Zodiac (Crime Fiction Film)""]], ""Robert Downey Jr. starred in Zodiac (Crime Fiction Film)""]}, 
		""type"": ""action"", 
		""sender"": ""user"", 
		""action_id"": ""kgwalk/choose_path""
	}, 
	...
]
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
 @inproceedings{moon-etal-2019-opendialkg,
    title = "{O}pen{D}ial{KG}: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs",
    author = "Moon, Seungwhan  and
      Shah, Pararth  and
      Kumar, Anuj  and
      Subba, Rajen",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1081",
    doi = "10.18653/v1/P19-1081",
    pages = "845--854",
}
```