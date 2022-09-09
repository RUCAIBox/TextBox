# Natural Questions

## Dataset

### Instruction

Paper: [Paper](https://aclanthology.org/Q19-1026.pdf)

Homepage: [Official](https://ai.google.com/research/NaturalQuestions/)

The Natural Questions corpus is a question answering dataset containing 307,373 training examples, 7,830 development examples, and 7,842 test examples. Each example is comprised of a google.com query and a corresponding Wikipedia page. Each Wikipedia page has a passage (or long answer) annotated on the page that answers the question and one or more short spans from the annotated passage containing the actual answer. The long and the short answer annotations can however be empty. If they are both empty, then there is no answer on the page at all. If the long answer annotation is non-empty, but the short answer annotation is empty, then the annotated passage answers the question but no explicit short answer could be found. Finally 1% of the documents have a passage annotated with a short answer that is “yes” or “no”, instead of a list of short spans.

### Overview

| Dataset           | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ----------------- | --------- | --------- | -------- | ------------------- | ------------------- |
| Natural Questions | $96,676$  | $10,693$  | $6,490$  | $9.0$               | $2.1$               |

### Data Sample

Input

```
when are hops added to the brewing process?
```

Short Answer

```
The boiling process
```

Long Answer

```
After mashing , the beer wort is boiled with hops ( and other flavourings if used ) in a large tank known as a " copper " or brew kettle – though historically the mash vessel was used and is still in some small breweries . The boiling process is where chemical reactions take place , including sterilization of the wort to remove unwanted bacteria , releasing of hop flavours , bitterness and aroma compounds through isomerization , stopping of enzymatic processes , precipitation of proteins , and concentration of the wort . Finally , the vapours produced during the boil volatilise off - flavours , including dimethyl sulfide precursors . The boil is conducted so that it is even and intense – a continuous " rolling boil " . The boil on average lasts between 45 and 90 minutes , depending on its intensity , the hop addition schedule , and volume of water the brewer expects to evaporate . At the end of the boil , solid particles in the hopped wort are separated out , usually in a vessel called a " whirlpool ".
```

## LeaderBoard

Descending order by EM.

| Model                                                  | EM     | Repository                                             | Generated Text |
| ------------------------------------------------------ | ------ | ------------------------------------------------------ | -------------- |
| [PaLM-540B](https://arxiv.org/pdf/2204.02311v3.pdf)    | $39.6$ | [Official](https://github.com/lucidrains/PaLM-pytorch) |                |
| [GLaM 62B/64E](https://arxiv.org/pdf/2112.06905v2.pdf) | $32.5$ |                                                        |                |
| [GPT-3 175B](https://arxiv.org/pdf/2005.14165v4.pdf)   | $29.9$ | [Official](https://github.com/openai/gpt-3)            |                |

## Citation

```
@article{nq,
    title = "Natural Questions: A Benchmark for Question Answering Research",
    author = "Kwiatkowski, Tom  and
      Palomaki, Jennimaria  and
      Redfield, Olivia  and
      Collins, Michael  and
      Parikh, Ankur  and
      Alberti, Chris  and
      Epstein, Danielle  and
      Polosukhin, Illia  and
      Devlin, Jacob  and
      Lee, Kenton  and
      Toutanova, Kristina  and
      Jones, Llion  and
      Kelcey, Matthew  and
      Chang, Ming-Wei  and
      Dai, Andrew M.  and
      Uszkoreit, Jakob  and
      Le, Quoc  and
      Petrov, Slav",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "7",
    year = "2019",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q19-1026",
    doi = "10.1162/tacl_a_00276",
    pages = "452--466",
}
```

