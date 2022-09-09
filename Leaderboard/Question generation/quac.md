# QuAC

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/D18-1241.pdf)

Homepage: [Official](https://quac.ai/)

Repository: [Official](https://github.com/deepnlp-cs599-usc/quac)

**Qu**estion **A**nswering in **C**ontext is a dataset for modeling, understanding, and participating in information seeking dialog. Data instances consist of an interactive dialog between two crowd workers: (1) a *student* who poses a sequence of freeform questions to learn as much as possible about a hidden Wikipedia text, and (2) a *teacher* who answers the questions by providing short excerpts (spans) from the text. QuAC introduces challenges not found in existing machine comprehension datasets: its questions are often more open-ended, unanswerable, or only meaningful within the dialog context.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| QuAC    | $83,568$  | $31,906$  | -        | $487.9$             | $12.5$              |

### Data Sample

```
{
  'dialogue_id': 'C_6abd2040a75d47168a9e4cca9ca3fed5_0',

  'wikipedia_page_title': 'Satchel Paige',

  'background': 'Leroy Robert "Satchel" Paige (July 7, 1906 - June 8, 1982) was an American Negro league baseball and Major League Baseball (MLB) pitcher who became a legend in his own lifetime by being known as perhaps the best pitcher in baseball history, by his longevity in the game, and by attracting record crowds wherever he pitched. Paige was a right-handed pitcher, and at age 42 in 1948, he was the oldest major league rookie while playing for the Cleveland Indians. He played with the St. Louis Browns until age 47, and represented them in the All-Star Game in 1952 and 1953.',

  'section_title': 'Chattanooga and Birmingham: 1926-29',

  'context': 'A former friend from the Mobile slums, Alex Herman, was the player/manager for the Chattanooga White Sox of the minor Negro Southern League. In 1926 he discovered Paige and offered to pay him $250 per month, of which Paige would collect $50 with the rest going to his mother. He also agreed to pay Lula Paige a $200 advance, and she agreed to the contract. The local newspapers--the Chattanooga News and Chattanooga Times--recognized from the beginning that Paige was special. In April 1926, shortly after his arrival, he recorded nine strikeouts over six innings against the Atlanta Black Crackers. Part way through the 1927 season, Paige\'s contract was sold to the Birmingham Black Barons of the major Negro National League (NNL). According to Paige\'s first memoir, his contract was for $450 per month, but in his second he said it was for $275. Pitching for the Black Barons, Paige threw hard but was wild and awkward. In his first big game in late June 1927, against the St. Louis Stars, Paige incited a brawl when his fastball hit the hand of St. Louis catcher Mitchell Murray. Murray then charged the mound and Paige raced for the dugout, but Murray flung his bat and struck Paige above the hip. The police were summoned, and the headline of the Birmingham Reporter proclaimed a "Near Riot." Paige improved and matured as a pitcher with help from his teammates, Sam Streeter and Harry Salmon, and his manager, Bill Gatewood. He finished the 1927 season 7-1 with 69 strikeouts and 26 walks in 89 1/3 innings. Over the next two seasons, Paige went 12-5 and 10-9 while recording 176 strikeouts in 1929. (Several sources credit his 1929 strikeout total as the all-time single-season record for the Negro leagues, though there is variation among the sources about the exact number of strikeouts.) On April 29 of that season he recorded 17 strikeouts in a game against the Cuban Stars, which exceeded what was then the major league record of 16 held by Noodles Hahn and Rube Waddell. Six days later he struck out 18 Nashville Elite Giants, a number that was tied in the white majors by Bob Feller in 1938. Due to his increased earning potential, Barons owner R. T. Jackson would "rent" Paige out to other ball clubs for a game or two to draw a decent crowd, with both Jackson and Paige taking a cut. CANNOTANSWER',

  'turn_ids': ['C_6abd2040a75d47168a9e4cca9ca3fed5_0_q#0', 'C_6abd2040a75d47168a9e4cca9ca3fed5_0_q#1', 'C_6abd2040a75d47168a9e4cca9ca3fed5_0_q#2', 'C_6abd2040a75d47168a9e4cca9ca3fed5_0_q#3', 'C_6abd2040a75d47168a9e4cca9ca3fed5_0_q#4', 'C_6abd2040a75d47168a9e4cca9ca3fed5_0_q#5', 'C_6abd2040a75d47168a9e4cca9ca3fed5_0_q#6', 'C_6abd2040a75d47168a9e4cca9ca3fed5_0_q#7'],

  'questions': ['what did he do in Chattanooga', 'how did he discover him', 'what position did he play', 'how did they help him', 'when did he go to Birmingham', 'how did he feel about this', 'how did he do with this team', 'What made him leave the team'],

  'followups': [0, 2, 0, 1, 0, 1, 0, 1],

  'yesnos': [2, 2, 2, 2, 2, 2, 2, 2]

  'answers': {
    'answer_starts': [
      [480, 39, 0, 67, 39],
      [2300, 2300, 2300],
      [848, 1023, 848, 848, 1298],
      [2300, 2300, 2300, 2300, 2300],
      [600, 600, 600, 634, 600],
      [2300, 2300, 2300],
      [939, 1431, 848, 848, 1514],
      [2106, 2106, 2165]
    ],
    'texts': [
      ['April 1926, shortly after his arrival, he recorded nine strikeouts over six innings against the Atlanta Black Crackers.', 'Alex Herman, was the player/manager for the Chattanooga White Sox of the minor Negro Southern League. In 1926 he discovered Paige', 'A former friend from the Mobile slums, Alex Herman, was the player/manager for the Chattanooga White Sox of the minor Negro Southern League.', 'manager for the Chattanooga White Sox of the minor Negro Southern League. In 1926 he discovered Paige and offered to pay him $250 per month,', 'Alex Herman, was the player/manager for the Chattanooga White Sox of the minor Negro Southern League. In 1926 he discovered Paige and offered to pay him $250 per month,'],
      ['CANNOTANSWER', 'CANNOTANSWER', 'CANNOTANSWER'],
      ['Pitching for the Black Barons,', 'fastball', 'Pitching for', 'Pitching', 'Paige improved and matured as a pitcher with help from his teammates,'], ['CANNOTANSWER', 'CANNOTANSWER', 'CANNOTANSWER', 'CANNOTANSWER', 'CANNOTANSWER'],
      ["Part way through the 1927 season, Paige's contract was sold to the Birmingham Black Barons", "Part way through the 1927 season, Paige's contract was sold to the Birmingham Black Barons", "Part way through the 1927 season, Paige's contract was sold to the Birmingham Black Barons", "Paige's contract was sold to the Birmingham Black Barons of the major Negro National League (NNL", "Part way through the 1927 season, Paige's contract was sold to the Birmingham Black Barons"], ['CANNOTANSWER', 'CANNOTANSWER', 'CANNOTANSWER'],
      ['game in late June 1927, against the St. Louis Stars, Paige incited a brawl when his fastball hit the hand of St. Louis catcher Mitchell Murray.', 'He finished the 1927 season 7-1 with 69 strikeouts and 26 walks in 89 1/3 innings.', 'Pitching for the Black Barons, Paige threw hard but was wild and awkward.', 'Pitching for the Black Barons, Paige threw hard but was wild and awkward.', 'Over the next two seasons, Paige went 12-5 and 10-9 while recording 176 strikeouts in 1929. ('],
      ['Due to his increased earning potential, Barons owner R. T. Jackson would "rent" Paige out to other ball clubs', 'Due to his increased earning potential, Barons owner R. T. Jackson would "rent" Paige out to other ball clubs for a game or two to draw a decent crowd,', 'Jackson would "rent" Paige out to other ball clubs for a game or two to draw a decent crowd, with both Jackson and Paige taking a cut.']
    ]
  },

  'orig_answers': {
    'answer_starts': [39, 2300, 1298, 2300, 600, 2300, 1514, 2165],
    'texts': ['Alex Herman, was the player/manager for the Chattanooga White Sox of the minor Negro Southern League. In 1926 he discovered Paige and offered to pay him $250 per month,', 'CANNOTANSWER', 'Paige improved and matured as a pitcher with help from his teammates,', 'CANNOTANSWER', "Part way through the 1927 season, Paige's contract was sold to the Birmingham Black Barons", 'CANNOTANSWER', 'Over the next two seasons, Paige went 12-5 and 10-9 while recording 176 strikeouts in 1929. (', 'Jackson would "rent" Paige out to other ball clubs for a game or two to draw a decent crowd, with both Jackson and Paige taking a cut.']
  },
}
```

## LeaderBoard

Descending order by METRIC.

| Model | METRIC | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
|       |        |            |                |
|       |        |            |                |
|       |        |            |                |

## Citation

```
@inproceedings{choi-etal-2018-quac,
    title = "{Q}u{AC}: Question Answering in Context",
    author = "Choi, Eunsol  and
      He, He  and
      Iyyer, Mohit  and
      Yatskar, Mark  and
      Yih, Wen-tau  and
      Choi, Yejin  and
      Liang, Percy  and
      Zettlemoyer, Luke",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1241",
    doi = "10.18653/v1/D18-1241",
    pages = "2174--2184",
}
```

 