# Movie Dialog

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1511.06931)

Homepage: [Official](https://research.fb.com/downloads/babi/)

Movie Dialog dataset (MDD) is designed to measure how well models can perform at goal and non-goal orientated dialog centered around the topic of movies (question answering, recommendation and discussion).

### Overview

| Dataset      | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------------ | --------- | --------- | -------- | ------------------- | ------------------- |
| Movie Dialog | $762,751$ | $8,216$   | $8,066$  | $126.9$             | $44.0$              |

### Data Sample

```
{
	'dialogue_turns': {
		'speaker': [0, 1, 0, 1, 0, 1], 
		'utterance': [
			"I really like Jaws, Bottle Rocket, Saving Private Ryan, Tommy Boy, The Muppet Movie, Face/Off, and Cool Hand Luke. I'm looking for a Documentary movie.", 
			'Beyond the Mat', 
			'Who is that directed by?', 
			'Barry W. Blaustein', 
			'I like Jon Fauer movies more. Do you know anything else?', 
			'Cinematographer Style'
		]
	}
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
 @inproceedings{md,
  author    = {Jesse Dodge and
               Andreea Gane and
               Xiang Zhang and
               Antoine Bordes and
               Sumit Chopra and
               Alexander H. Miller and
               Arthur Szlam and
               Jason Weston},
  title     = {Evaluating Prerequisite Qualities for Learning End-to-End Dialog Systems},
  booktitle = {4th International Conference on Learning Representations, {ICLR} 2016},
  year      = {2016},
  url={http://arxiv.org/abs/1511.06931},
}
```