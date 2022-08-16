# NewsQA

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/W17-2623.pdf)

Homepage: [Official](https://www.microsoft.com/en-us/research/project/newsqa-dataset/)

The NewsQA dataset is a crowd-sourced machine reading comprehension dataset of 120,000 question-answer pairs.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| NewsQA  | $97,850$  | $5,486$   | $5,396$  | $726.8$             | $5.0$               |

### Data Sample

```
{'storyId': './cnn/stories/42d01e187213e86f5fe617fe32e716ff7fa3afc4.story',
 'text': 'NEW DELHI, India (CNN) -- A high court in northern India on Friday acquitted a wealthy businessman facing the death sentence for the killing of a teen in a case dubbed "the house of horrors."\n\n\n\nMoninder Singh Pandher was sentenced to death by a lower court in February.\n\n\n\nThe teen was one of 19 victims -- children and young women -- in one of the most gruesome serial killings in India in recent years.\n\n\n\nThe Allahabad high court has acquitted Moninder Singh Pandher, his lawyer Sikandar B. Kochar told CNN.\n\n\n\nPandher and his domestic employee Surinder Koli were sentenced to death in February by a lower court for the rape and murder of the 14-year-old.\n\n\n\nThe high court upheld Koli\'s death sentence, Kochar said.\n\n\n\nThe two were arrested two years ago after body parts packed in plastic bags were found near their home in Noida, a New Delhi suburb. Their home was later dubbed a "house of horrors" by the Indian media.\n\n\n\nPandher was not named a main suspect by investigators initially, but was summoned as co-accused during the trial, Kochar said.\n\n\n\nKochar said his client was in Australia when the teen was raped and killed.\n\n\n\nPandher faces trial in the remaining 18 killings and could remain in custody, the attorney said.',
 'type': 'train',
 'questions': {'q': ['What was the amount of children murdered?',
   'When was Pandher sentenced to death?',
   'The court aquitted Moninder Singh Pandher of what crime?',
   'who was acquitted',
   'who was sentenced',
   'What was Moninder Singh Pandher acquitted for?',
   'Who was sentenced to death in February?',
   'how many people died',
   'How many children and young women were murdered?'],
  'isAnswerAbsent': [0, 0, 0, 0, 0, 0, 0, 0, 0],
  'isQuestionBad': [0, 0, 0, 0, 0, 0, 0, 0, 0],
  'consensus': [{'s': 294, 'e': 297, 'badQuestion': False, 'noAnswer': False},
   {'s': 261, 'e': 271, 'badQuestion': False, 'noAnswer': False},
   {'s': 624, 'e': 640, 'badQuestion': False, 'noAnswer': False},
   {'s': 195, 'e': 218, 'badQuestion': False, 'noAnswer': False},
   {'s': 195, 'e': 218, 'badQuestion': False, 'noAnswer': False},
   {'s': 129, 'e': 151, 'badQuestion': False, 'noAnswer': False},
   {'s': 195, 'e': 218, 'badQuestion': False, 'noAnswer': False},
   {'s': 294, 'e': 297, 'badQuestion': False, 'noAnswer': False},
   {'s': 294, 'e': 297, 'badQuestion': False, 'noAnswer': False}],
  'answers': [{'sourcerAnswers': [{'s': [294],
      'e': [297],
      'badQuestion': [False],
      'noAnswer': [False]},
     {'s': [0], 'e': [0], 'badQuestion': [False], 'noAnswer': [True]},
     {'s': [0], 'e': [0], 'badQuestion': [False], 'noAnswer': [True]}]},
   {'sourcerAnswers': [{'s': [261],
      'e': [271],
      'badQuestion': [False],
      'noAnswer': [False]},
     {'s': [258], 'e': [271], 'badQuestion': [False], 'noAnswer': [False]},
     {'s': [261], 'e': [271], 'badQuestion': [False], 'noAnswer': [False]}]},
   {'sourcerAnswers': [{'s': [26],
      'e': [33],
      'badQuestion': [False],
      'noAnswer': [False]},
     {'s': [0], 'e': [0], 'badQuestion': [False], 'noAnswer': [True]},
     {'s': [624], 'e': [640], 'badQuestion': [False], 'noAnswer': [False]}]},
   {'sourcerAnswers': [{'s': [195],
      'e': [218],
      'badQuestion': [False],
      'noAnswer': [False]},
     {'s': [195], 'e': [218], 'badQuestion': [False], 'noAnswer': [False]}]},
   {'sourcerAnswers': [{'s': [0],
      'e': [0],
      'badQuestion': [False],
      'noAnswer': [True]},
     {'s': [195, 232],
      'e': [218, 271],
      'badQuestion': [False, False],
      'noAnswer': [False, False]},
     {'s': [0], 'e': [0], 'badQuestion': [False], 'noAnswer': [True]}]},
   {'sourcerAnswers': [{'s': [129],
      'e': [192],
      'badQuestion': [False],
      'noAnswer': [False]},
     {'s': [129], 'e': [151], 'badQuestion': [False], 'noAnswer': [False]},
     {'s': [133], 'e': [151], 'badQuestion': [False], 'noAnswer': [False]}]},
   {'sourcerAnswers': [{'s': [195],
      'e': [218],
      'badQuestion': [False],
      'noAnswer': [False]},
     {'s': [195], 'e': [218], 'badQuestion': [False], 'noAnswer': [False]}]},
   {'sourcerAnswers': [{'s': [294],
      'e': [297],
      'badQuestion': [False],
      'noAnswer': [False]},
     {'s': [294], 'e': [297], 'badQuestion': [False], 'noAnswer': [False]}]},
   {'sourcerAnswers': [{'s': [294],
      'e': [297],
      'badQuestion': [False],
      'noAnswer': [False]},
     {'s': [294], 'e': [297], 'badQuestion': [False], 'noAnswer': [False]}]}],
  'validated_answers': [{'s': [0, 294],
    'e': [0, 297],
    'badQuestion': [False, False],
    'noAnswer': [True, False],
    'count': [1, 2]},
   {'s': [], 'e': [], 'badQuestion': [], 'noAnswer': [], 'count': []},
   {'s': [624],
    'e': [640],
    'badQuestion': [False],
    'noAnswer': [False],
    'count': [2]},
   {'s': [], 'e': [], 'badQuestion': [], 'noAnswer': [], 'count': []},
   {'s': [195],
    'e': [218],
    'badQuestion': [False],
    'noAnswer': [False],
    'count': [2]},
   {'s': [129],
    'e': [151],
    'badQuestion': [False],
    'noAnswer': [False],
    'count': [2]},
   {'s': [], 'e': [], 'badQuestion': [], 'noAnswer': [], 'count': []},
   {'s': [], 'e': [], 'badQuestion': [], 'noAnswer': [], 'count': []},
   {'s': [], 'e': [], 'badQuestion': [], 'noAnswer': [], 'count': []}]}}
```

## LeaderBoard

Descending order by F1.

| Model                                                | F1     | EM     | Repository                                                | Generated Text |
| ---------------------------------------------------- | ------ | ------ | --------------------------------------------------------- | -------------- |
| [ SpanBERT](https://arxiv.org/pdf/1907.10529v3.pdf)  | $73.6$ |        | [Official](https://github.com/facebookresearch/SpanBERT)  |                |
| [ LinkBERT](https://arxiv.org/pdf/2203.15827v1.pdf)  | $72.6$ |        | [Official](https://github.com/michiyasunaga/LinkBERT)     |                |
| [ DecaProp](https://arxiv.org/pdf/1811.04210v2.pdf)  | $66.3$ | $53.1$ | [Official](https://github.com/vanzytay/NIPS2018_DECAPROP) |                |
| [ AMANDA](https://arxiv.org/pdf/1801.08290v1.pdf)    | $63.7$ | $48.4$ | [Official](https://github.com/nusnlp/amanda)              |                |
| [ MINIMAL](https://arxiv.org/pdf/1805.08092v1.pdf)   | $63.2$ | $50.1$ |                                                           |                |
| [ FastQAExt](https://arxiv.org/pdf/1703.04816v3.pdf) | $56.1$ | $43.7$ |                                                           |                |

## Citation

```
@inproceedings{trischler-etal-2017-newsqa,
    title = "{N}ews{QA}: A Machine Comprehension Dataset",
    author = "Trischler, Adam  and
      Wang, Tong  and
      Yuan, Xingdi  and
      Harris, Justin  and
      Sordoni, Alessandro  and
      Bachman, Philip  and
      Suleman, Kaheer",
    booktitle = "Proceedings of the 2nd Workshop on Representation Learning for {NLP}",
    month = aug,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-2623",
    doi = "10.18653/v1/W17-2623",
    pages = "191--200",
}
```

