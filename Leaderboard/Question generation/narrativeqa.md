# NarrativeQA

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1712.07040)

Repository: [Official](https://github.com/deepmind/narrativeqa)

NarrativeQA is an English-lanaguage dataset of stories and corresponding questions designed to test reading comprehension, especially on long documents.

### Overview

| Dataset     | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ----------- | --------- | --------- | -------- | ------------------- | ------------------- |
| NattativeQA | $65,494$  | $6,922$   | $21,114$ | $584.1$             | $4.2$               |

### Data Sample

```
{
    "document": {
        "id": "23jncj2n3534563110",
        "kind": "movie",
        "url": "https://www.imsdb.com/Movie%20Scripts/Name%20of%20Movie.html",
        "file_size": 80473,
        "word_count": 41000,
        "start": "MOVIE screenplay by",
        "end": ". THE END",
        "summary": {
            "text": "Joe Bloggs begins his journey exploring...",
            "tokens": ["Joe", "Bloggs", "begins", "his", "journey", "exploring",...],
            "url": "http://en.wikipedia.org/wiki/Name_of_Movie",
            "title": "Name of Movie (film)"
        },
        "text": "MOVIE screenplay by John Doe\nSCENE 1..."
    },
    "question": {
        "text": "Where does Joe Bloggs live?",
        "tokens": ["Where", "does", "Joe", "Bloggs", "live", "?"],
    },
    "answers": [
        {"text": "At home", "tokens": ["At", "home"]},
        {"text": "His house", "tokens": ["His", "house"]}
    ]
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
@article{kocisky-etal-2018-narrativeqa,
    title = "The {N}arrative{QA} Reading Comprehension Challenge",
    author = "Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
      Schwarz, Jonathan  and
      Blunsom, Phil  and
      Dyer, Chris  and
      Hermann, Karl Moritz  and
      Melis, G{\'a}bor  and
      Grefenstette, Edward",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "6",
    year = "2018",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q18-1023",
    doi = "10.1162/tacl_a_00023",
    pages = "317--328",
}
```

 