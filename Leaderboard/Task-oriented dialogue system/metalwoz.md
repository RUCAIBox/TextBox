# MetaLWOZ

## Dataset

### Instruction

Paper: [Paper](http://workshop.colips.org/dstc7/dstc8/DTSC8_multidomain_task_proposal.pdf)

Homepage: [Official](https://www.microsoft.com/en-us/research/project/metalwoz/)

Collected by leveraging background knowledge from a larger, more highly represented dialogue source.

### Overview

| Dataset  | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | -------- | ------------------- | ------------------- |
| MetaLWOZ | $176,073$ | $17,912$  | -        | $45.6$              | $8.0$               |

### Data Sample

Full Dialog

```
Domain: Ski
User Task: You want to know if there are good ski hills an
hour’s drive from your current location.
Bot Task: Tell the user that there are no ski hills in their
immediate location.
Bot: Hello how may I help you?
User: Is there any good ski hills an hour’s drive from my
current location?
Bot: I’m sorry to inform you that there are no ski hills in your
immediate location
User: Can you help me find the nearest?
Bot: Absolutely! It looks like you’re about 3 hours away from
Bear Mountain. That seems to be the closest.
User: Hmm.. sounds good
Bot: Alright! I can help you get your lift tickets now!When
will you be going?
User: Awesome! please get me a ticket for 10pax
Bot: You’ve got it. Anything else I can help you with?
User: None. Thanks again!
Bot: No problem!
```

Example Input

```
dialog history = Hello how may I help you?; Is there any good ski hills an hour’s drive from my current location?; I’m sorry to inform you that there are no ski hills in your immediate location
```

Example Output

```
user response = Can you help me find the nearest?
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
@inproceedings{metalwoz,
	author = {Lee, Sungjin and Schulz, Hannes and Atkinson, Adam and Gao, Jianfeng and Suleman, Kaheer and El 	Asri, Layla and Adada, Mahmoud and Huang, Minlie and Sharma, Shikhar and Tay, Wendy and Li, Xiujun},
	title = {Multi-Domain Task-Completion Dialog Challenge},
	booktitle = {Dialog System Technology Challenges},
	volume={8},
	number={9},
	url={http://workshop.colips.org/dstc7/dstc8/DTSC8_multidomain_task_proposal.pdf},
	year = {2019},
}
```

