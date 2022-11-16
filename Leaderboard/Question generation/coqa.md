# CoQA

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1808.07042)

Homepage: [Official](https://stanfordnlp.github.io/coqa/)

CoQA is a large-scale dataset for building Conversational Question Answering systems. The goal of the CoQA challenge is to measure the ability of machines to understand a text passage and answer a series of interconnected questions that appear in a conversation. CoQA is pronounced as coca.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| CoQA    | $107,286$ | $31,621$  | -        | $349.4$             | $2.6$               |

### Data Sample

Input

```
'white [X_SEP] Once upon a time, in a barn near a farm house, there lived a little white kitten named Cotton. Cotton lived high up in a nice warm place above the barn where all of the farmer\'s horses slept. But Cotton wasn\'t alone in her little home above the barn, oh no. She shared her hay bed with her mommy and 5 other sisters. All of her sisters were cute and fluffy, like Cotton. But she was the only white one in the bunch. The rest of her sisters were all orange with beautiful white tiger stripes like Cotton\'s mommy. Being different made Cotton quite sad. She often wished she looked like the rest of her family. So one day, when Cotton found a can of the old farmer\'s orange paint, she used it to paint herself like them. When her mommy and sisters found her they started laughing.\n\n"What are you doing, Cotton?!"\n\n"I only wanted to be more like you".\n\nCotton\'s mommy rubbed her face on Cotton\'s and said "Oh Cotton, but your fur is so pretty and special, like you. We would never want you to be any other way". And with that, Cotton\'s mommy picked her up and dropped her into a big bucket of water. When Cotton came out she was herself again. Her sisters licked her face until Cotton\'s fur was all all dry.\n\n"Don\'t ever do that again, Cotton!" they all cried. "Next time you might mess up that pretty white fur of yours and we wouldn\'t want that!"\n\nThen Cotton thought, "I change my mind. I like being special".'
```

Output

```
'What color was Cotton?'
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
@article{coqa,
    title = "{C}o{QA}: A Conversational Question Answering Challenge",
    author = "Reddy, Siva  and
      Chen, Danqi  and
      Manning, Christopher D.",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "7",
    year = "2019",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q19-1016",
    doi = "10.1162/tacl_a_00266",
    pages = "249--266",
}
```

