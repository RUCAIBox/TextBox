# SGD

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1909.05855)

Repository: [Official](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)

The Schema-Guided Dialogue (SGD) dataset consists of over 20k annotated multi-domain, task-oriented conversations between a human and a virtual assistant. These conversations involve interactions with services and APIs spanning 20 domains, ranging from banks and events to media, calendar, travel, and weather. For most of these domains, the dataset contains multiple different APIs, many of which have overlapping functionalities but different interfaces, which reflects common real-world scenarios. The wide range of available annotations can be used for intent prediction, slot filling, dialogue state tracking, policy imitation learning, language generation, user simulation learning, among other tasks in large-scale virtual assistants. Besides these, the dataset has unseen domains and services in the evaluation set to quantify the performance in zero-shot or few shot settings.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| SGD     | $494,946$ | $73,089$  | -        | $120.8$             | $12.5$              |

### Data Sample

Input

```
"song_name | Carnivore [SEP] playback_device | TV [X_SEP] Yes, I would. [X_SEP] I need a movie to watch. [SEP] I have 10 movies. What about Searching for Sugar Man rated 8.2. [SEP] What about a drama genre movie? [SEP] What about Dogman rated 7.3? [SEP] Sure, I'd like some songs searched for now. [SEP] I have 10 songs. What about Despacito by Karolina Protsenko from the album My Dream. [SEP] What type of music it that? [SEP] That's a pop song. [SEP] What year is it from? [SEP] The year is 2018. [SEP] What other items do you have? [SEP] What about your thoughts on Carnivore by Starset in the album Transmissions? [SEP] Yes, that is it. [SEP] Do you want to play this song?"
```

Output

```
['Please confirm the song Carnivore on tv.']
```

## LeaderBoard

Descending order by METEOR.

| Model                                          | METEOR | Repository | Generated Text |
| ---------------------------------------------- | ------ | ---------- | -------------- |
| [T5](https://arxiv.org/pdf/2102.01672v3.pdf)   | $33.1$ |            |                |
| [BART](https://arxiv.org/pdf/2102.01672v3.pdf) | $8.9$  |            |                |
|                                                |        |            |                |

## Citation

```
 @inproceedings{sgd, 
	title={Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset}, 
	volume={34}, 
	url={https://ojs.aaai.org/index.php/AAAI/article/view/6394}, 
	DOI={10.1609/aaai.v34i05.6394}, 
	number={05}, 
	journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
	author={Rastogi, Abhinav and Zang, Xiaoxue and Sunkara, Srinivas and Gupta, Raghav and Khaitan, Pranav}, 
	year={2020}, 
	month={Apr.}, 
	pages={8689-8696}
}
```