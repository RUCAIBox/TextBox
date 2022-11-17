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

Input

```
'cast | Kristen Bell as Anna, the 18-year-old Princess of Arendelle and Elsa\'s younger sister, Livvy Stubenrauch as 5-year-old Anna, Katie Lopez as 5-year-old Anna (singing), Agatha Lee Monn as 9-year-old Anna. Idina Menzel as Elsa, the 21-year-old Snow Queen of Arendelle and Anna\'s elder sister, Eva Bella as 8-year-old Elsa, Spencer Lacey Ganus as 12-year-old Elsa. Jonathan Groff as Kristoff, an iceman who is accompanied by a reindeer named Sven, Tyree Brown as 8-year-old Kristoff. [SEP] critical_response | the best animated musical to come out of Disney since the tragic death of lyricist Howard Ashman, whose work on The Little Mermaid and Beauty and the Beast helped build the studio\'s modern animated division into what it is today. while it lags the tiniest bit on its way to the conclusion, the script... really delivers; it offers characters to care about, along with some nifty twists and surprises along the way. You can practically see the Broadway musical Frozen is destined to become while watching Disney\'s 3D animated princess tale. a great big snowy pleasure with an emotionally gripping core, brilliant Broadway-style songs and a crafty plot. Its first and third acts are better than the jokey middle, but this is the rare example of a Walt Disney Animation Studios effort that reaches as deep as a Pixar film. Frozen is both a declaration of Disney\'s renewed cultural relevance and a reaffirmation of Disney coming to terms with its own legacy and its own identity. It\'s also a just plain terrific bit of family entertainment. [SEP] director | Chris Buck, Jennifer Lee [SEP] genre | Comedy, Adventure, Animation [SEP] introduction | Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures. It is the 53rd Disney animated feature film. Inspired by Hans Christian Andersen\'s fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged iceman, his loyal pet reindeer, and a nai ve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter. [SEP] movieName | Frozen [SEP] rating | Rotten Tomatoes: 89%. Metacritics: 74/100. CinemaScore: A+. [SEP] year | 2013 [X_SEP] Hello? [SEP] Hi'
```

Output

```
'What movie are you reading about over there? :)'
```

## LeaderBoard

Descending order by F1.

| Model                                                | PPL    | F1     | Repository                                          | Generated Text |
| ---------------------------------------------------- | ------ | ------ | --------------------------------------------------- | -------------- |
| [KnowledGPT](https://arxiv.org/pdf/2010.08824v1.pdf) | $20.6$ | $13.5$ | [Official](https://github.com/zhaoxlpku/KnowledGPT) |                |
| [GPT-2](https://arxiv.org/pdf/2010.08824v1.pdf)      | $18.6$ | $10.8$ |                                                     |                |
| [DRD](https://arxiv.org/pdf/2010.08824v1.pdf)        | $46.1$ | $10.8$ |                                                     |                |
| [ITDD](https://arxiv.org/pdf/2010.08824v1.pdf)       | $26.0$ | $10.4$ |                                                     |                |
| [TMN](https://arxiv.org/pdf/2010.08824v1.pdf)        | $75.2$ | $9.9$  |                                                     |                |

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