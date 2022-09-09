# TaskMaster

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/D19-1459.pdf)

Homepage: [Official](https://research.google/tools/datasets/taskmaster-1/)

Repository: [Official](https://github.com/google-research-datasets/Taskmaster)

Taskmaster-1 is a dialog dataset consisting of 13,215 task-based dialogs in English, including 5,507 spoken and 7,708 written dialogs created with two distinct procedures. Each conversation falls into one of six domains: ordering pizza, creating auto repair appointments, setting up ride service, ordering movie tickets, ordering coffee drinks and making restaurant reservations.

### Overview

| Dataset    | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ---------- | --------- | --------- | -------- | ------------------- | ------------------- |
| TaskMaster | $249,664$ | $20,680$  | -        | $95.6$              | $12.0$              |

### Data Sample

```
{
    "conversation_id": "dlg-00055f4e-4a46-48bf-8d99-4e477663eb23",
    "instruction_id": "restaurant-table-2",
    "utterances": [
      {
        "index": 0,
        "speaker": "USER",
        "text": "Hi, I'm looking to book a table for Korean food."
      },
      {
        "index": 1,
        "speaker": "ASSISTANT",
        "text": "Ok, what area are you thinking about?"
      },
      {
        "index": 2,
        "speaker": "USER",
        "text": "Somewhere in Southern NYC, maybe the East Village?",
        "segments": [
          {
            "start_index": 13,
            "end_index": 49,
            "text": "Southern NYC, maybe the East Village",
            "annotations": [
              {
                "name": "restaurant_reservation.location.restaurant.accept"
              }
            ]
          },
          {
            "start_index": 13,
            "end_index": 25,
            "text": "Southern NYC",
            "annotations": [
              {
                "name": "restaurant_reservation.location.restaurant.accept"
              }
            ]
          }
        ]
      },
      {
        "index": 3,
        "speaker": "ASSISTANT",
        "text": "Ok, great.  There's Thursday Kitchen, it has great reviews.",
        "segments": [
          {
            "start_index": 20,
            "end_index": 35,
            "text": "Thursday Kitche",
            "annotations": [
              {
                "name": "restaurant_reservation.name.restaurant.reject"
              }
            ]
          }
        ]
      },
      {
        "index": 4,
        "speaker": "USER",
        "text": "That's great. So I need a table for tonight at 7 pm for 8 people. We don't want to sit at the bar, but anywhere else is fine.",
        "segments": [
          {
            "start_index": 26,
            "end_index": 31,
            "text": "table",
            "annotations": [
              {
                "name": "restaurant_reservation.type.seating"
              }
            ]
          },
          {
            "start_index": 47,
            "end_index": 51,
            "text": "7 pm",
            "annotations": [
              {
                "name": "restaurant_reservation.time.reservation"
              },
              {
                "name": "restaurant_reservation.time.reservation"
              }
            ]
          },
          {
            "start_index": 56,
            "end_index": 57,
            "text": "8",
            "annotations": [
              {
                "name": "restaurant_reservation.num.guests"
              },
              {
                "name": "restaurant_reservation.num.guests"
              }
            ]
          },
          {
            "start_index": 87,
            "end_index": 98,
            "text": "at the bar,",
            "annotations": [
              {
                "name": "restaurant_reservation.type.seating"
              }
            ]
          }
        ]
      },
      {
        "index": 5,
        "speaker": "ASSISTANT",
        "text": "They don't have any availability for 7 pm.",
        "segments": [
          {
            "start_index": 37,
            "end_index": 41,
            "text": "7 pm",
            "annotations": [
              {
                "name": "restaurant_reservation.time.reservation"
              }
            ]
          },
          {
            "start_index": 37,
            "end_index": 42,
            "text": "7 pm.",
            "annotations": [
              {
                "name": "restaurant_reservation.time.reservation.reject"
              }
            ]
          }
        ]
      },
      {
        "index": 6,
        "speaker": "USER",
        "text": "What times are available?"
      },
      {
        "index": 7,
        "speaker": "ASSISTANT",
        "text": "5 or 8.",
        "segments": [
          {
            "start_index": 0,
            "end_index": 1,
            "text": "5",
            "annotations": [
              {
                "name": "restaurant_reservation.time.reservation"
              },
              {
                "name": "restaurant_reservation.time.reservation"
              }
            ]
          },
          {
            "start_index": 5,
            "end_index": 6,
            "text": "8",
            "annotations": [
              {
                "name": "restaurant_reservation.time.reservation"
              },
              {
                "name": "restaurant_reservation.time.reservation"
              }
            ]
          }
        ]
      },
      {
        "index": 8,
        "speaker": "USER",
        "text": "Yikes, we can't do those times."
      },
      {
        "index": 9,
        "speaker": "ASSISTANT",
        "text": "Ok, do you have a second choice?"
      },
      {
        "index": 10,
        "speaker": "USER",
        "text": "Let me check."
      },
      {
        "index": 11,
        "speaker": "ASSISTANT",
        "text": "Ok."
      },
      {
        "index": 12,
        "speaker": "USER",
        "text": "Lets try Boka, are they free for 8 people at 7?",
        "segments": [
          {
            "start_index": 9,
            "end_index": 13,
            "text": "Boka",
            "annotations": [
              {
                "name": "restaurant_reservation.name.restaurant.accept"
              },
              {
                "name": "restaurant_reservation.name.restaurant.accept"
              }
            ]
          },
          {
            "start_index": 33,
            "end_index": 34,
            "text": "8",
            "annotations": [
              {
                "name": "restaurant_reservation.num.guests.accept"
              },
              {
                "name": "restaurant_reservation.num.guests.accept"
              }
            ]
          },
          {
            "start_index": 45,
            "end_index": 46,
            "text": "7",
            "annotations": [
              {
                "name": "restaurant_reservation.time.reservation.accept"
              },
              {
                "name": "restaurant_reservation.time.reservation.accept"
              }
            ]
          }
        ]
      },
      {
        "index": 13,
        "speaker": "ASSISTANT",
        "text": "Yes."
      },
      {
        "index": 14,
        "speaker": "USER",
        "text": "Great, let's book that."
      },
      {
        "index": 15,
        "speaker": "ASSISTANT",
        "text": "Ok great, are there any other requests?"
      },
      {
        "index": 16,
        "speaker": "USER",
        "text": "No, that's it, just book."
      },
      {
        "index": 17,
        "speaker": "ASSISTANT",
        "text": "Great, should I use your account you have open with them?"
      },
      {
        "index": 18,
        "speaker": "USER",
        "text": "Yes please."
      },
      {
        "index": 19,
        "speaker": "ASSISTANT",
        "text": "Great. You will get a confirmation to your phone soon."
      }
    ]
}
```

## LeaderBoard

Descending order by BLEU.

| Model                                                   | BLEU   | Repository | Generated Text |
| ------------------------------------------------------- | ------ | ---------- | -------------- |
| [Transformer](https://aclanthology.org/D19-1459.pdf)    | $6.11$ |            |                |
| [LSTM-attention](https://aclanthology.org/D19-1459.pdf) | $5.12$ |            |                |
| [Convolution](https://aclanthology.org/D19-1459.pdf)    | $5.09$ |            |                |
| [LSTM](https://aclanthology.org/D19-1459.pdf)           | $4.45$ |            |                |
| [4-gram](https://aclanthology.org/D19-1459.pdf)         | $0.21$ |            |                |
| [3-gram](https://aclanthology.org/D19-1459.pdf)         | $0.20$ |            |                |

## Citation

```
 @inproceedings{taskmaster,
    title = "Taskmaster-1: Toward a Realistic and Diverse Dialog Dataset",
    author = "Byrne, Bill  and
      Krishnamoorthi, Karthik  and
      Sankar, Chinnadhurai  and
      Neelakantan, Arvind  and
      Goodrich, Ben  and
      Duckworth, Daniel  and
      Yavuz, Semih  and
      Dubey, Amit  and
      Kim, Kyu-Young  and
      Cedilnik, Andy",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1459",
    doi = "10.18653/v1/D19-1459",
    pages = "4516--4525",
}
```