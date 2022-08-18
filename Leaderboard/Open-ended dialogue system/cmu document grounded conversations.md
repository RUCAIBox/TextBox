# CMU Document Grounded Conversations

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/D18-1076.pdf)

Repository: [Official](https://github.com/festvox/datasets-CMU_DoG)

In this dataset the specified documents were Wikipedia articles about popular movies. The dataset contains 4112 conversations with an average of 21.43 turns per conversation. This positions this dataset to not only provide a relevant chat history while generating responses but also provide a source of information that the models could use.

### Overview

| Dataset                             | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ----------------------------------- | --------- | --------- | -------- | ------------------- | ------------------- |
| CMU Document Grounded Conversations | $82,818$  | $5,555$   | $14,510$ | $433.0$             | $12.2$              |

### Data Sample

```
{
  "date": "2018-03-01T00:11:05.970Z", 
  "history": [
    {
      "docIdx": 0, 
      "text": "Hey there hows it going! You like catch me if you can as much as i do?", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:11:35.166Z"
    }, 
    {
      "docIdx": 0, 
      "text": "Opps I meant means girls!", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:11:58.270Z"
    }, 
    {
      "docIdx": 0, 
      "text": "Oh, Mean Girls? It's a great movie. Do you like Lindsay Lohan's role as Cady Heron?", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:12:31.151Z"
    }, 
    {
      "docIdx": 0, 
      "text": "Isn't Lindsey like the best female actress of all time or what?", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:12:40.294Z"
    }, 
    {
      "docIdx": 0, 
      "text": "Yeah thats here name in the movie ", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:12:59.535Z"
    }, 
    {
      "docIdx": 0, 
      "text": "I think Rachel McAdams had an even\n better role as Regina George however!", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:13:23.996Z"
    }, 
    {
      "docIdx": 0, 
      "text": "Would you agree?", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:13:32.731Z"
    }, 
    {
      "docIdx": 0, 
      "text": "Racheal Adams is wonderful as well!!!", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:13:39.223Z"
    }, 
    {
      "docIdx": 0, 
      "text": "\nbut i also like Regina George as well so its hard to pick to be honest", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:13:56.681Z"
    }, 
    {
      "docIdx": 0, 
      "text": "Well, Regina George was played by Rachel McAdams. No wonder it's hard for you to pick in that case!", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:14:30.943Z"
    }, 
    {
      "docIdx": 0, 
      "text": "Did you know that tina fey wrote this movie?", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:14:38.692Z"
    }, 
    {
      "docIdx": 1, 
      "text": "I did! She really delivered a knockout in Mean Girls.", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:15:05.018Z"
    }, 
    {
      "docIdx": 1, 
      "text": "What was your favorite scene in Mean Girls?", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:15:19.572Z"
    }, 
    {
      "docIdx": 1, 
      "text": "I personally like the scene where Cady met the Plastics.", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:15:52.148Z"
    }, 
    {
      "docIdx": 1, 
      "text": "I love the revenge plot the best I think", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:16:21.787Z"
    }, 
    {
      "docIdx": 1, 
      "text": "Oh yeah! The plan of revenge against Regina? That was awesome!", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:16:49.171Z"
    }, 
    {
      "docIdx": 1, 
      "text": "Did you know that Mean Girls was partially based on a book?", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:17:19.022Z"
    }, 
    {
      "docIdx": 1, 
      "text": "Was it really? I am not surprised at all.", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:17:37.013Z"
    }, 
    {
      "docIdx": 1, 
      "text": "Yeah! It was based on the book Queen Bees and Wannabes. Does that sound like your kind of book?", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:18:20.246Z"
    }, 
    {
      "docIdx": 2, 
      "text": "Oh no I am alergic to bees", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:18:50.132Z"
    }, 
    {
      "docIdx": 2, 
      "text": "Haha! Good one.", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:19:18.496Z"
    }, 
    {
      "docIdx": 2, 
      "text": "The burn book was such a designed to fail method", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:19:22.821Z"
    }, 
    {
      "docIdx": 2, 
      "text": "but you knew it was gonna cause funny drama haha", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:19:36.356Z"
    }, 
    {
      "docIdx": 2, 
      "text": "The drama the burn book caused was serious! ", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:20:22.884Z"
    }, 
    {
      "docIdx": 2, 
      "text": "A math teacher got defamed as a drug dealer because of it.", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:20:47.325Z"
    }, 
    {
      "docIdx": 3, 
      "text": "Regina getting hit by that bus was a really intenese scene", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:21:45.226Z"
    }, 
    {
      "docIdx": 3, 
      "text": "It was. Can you believe Cady took all the blame for the Burn Book?", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:22:12.592Z"
    }, 
    {
      "docIdx": 3, 
      "text": "i still can't believe she still did that ", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:22:31.107Z"
    }, 
    {
      "docIdx": 3, 
      "text": "She felt really guiltiy I guess but still", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:22:47.942Z"
    }, 
    {
      "docIdx": 3, 
      "text": "It did work out in the end though, wouldn't you agree? She got elected as the Queen of the Spring Fling dance!", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:23:03.906Z"
    }, 
    {
      "docIdx": 3, 
      "text": "yeah I guess thats always worth it, and a truce was made as well", 
      "uid": "user2", 
      "utcTimestamp": "2018-03-01T00:23:48.686Z"
    }, 
    {
      "docIdx": 3, 
      "text": "Yeah. Everyone made up with the Plastics. Alls well that ends well!", 
      "uid": "user1", 
      "utcTimestamp": "2018-03-01T00:24:12.448Z"
    }
  ], 
  "rating": 2, 
  "status": 1, 
  "uid1LogInTime": "2018-03-01T00:11:05.970Z", 
  "uid1LogOutTime": "2018-03-01T00:24:17.582Z", 
  "uid1response": {
    "feedback": "The other user had poor grammar and spelling", 
    "response": [
      1, 
      2, 
      3, 
      4
    ], 
    "type": "finish"
  }, 
  "uid2LogInTime": "2018-03-01T00:11:06.106Z", 
  "uid2LogOutTime": "2018-03-01T00:24:23.395Z", 
  "uid2response": {
    "feedback": null, 
    "response": [
      1, 
      2, 
      3, 
      4
    ], 
    "type": "finish"
  }, 
  "user1_id": "USR1906", 
  "user2_id": "USR3118", 
  "whoSawDoc": [
    "user1", 
    "user2"
  ], 
  "wikiDocumentIdx": 11
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
@inproceedings{zhou-etal-2018-dataset,
    title = "A Dataset for Document Grounded Conversations",
    author = "Zhou, Kangyan  and
      Prabhumoye, Shrimai  and
      Black, Alan W",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1076",
    doi = "10.18653/v1/D18-1076",
    pages = "708--713",
} 
```