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
'who got the first nobel prize in physics'
```

Output

```
'Wilhelm Conrad Röntgen'
```


## LeaderBoard

Descending order by EM.

| Model                                                  | EM     | Repository                                                   | Generated Text |
| ------------------------------------------------------ | ------ | ------------------------------------------------------------ | -------------- |
| [Atlas](https://arxiv.org/pdf/2208.03299v2.pdf)        | $60.4$ |                                                              |                |
| [R2-D2](https://arxiv.org/pdf/2109.03502v1.pdf)        | $59.9$ | [Official](https://github.com/KNOT-FIT-BUT/R2-D2)            |                |
| [FiD-KD](https://arxiv.org/pdf/2007.01282v2.pdf)       | $54.7$ |                                                              |                |
| [EMDR$^2$](https://arxiv.org/pdf/2106.05346v2.pdf)     | $52.5$ | [Official](https://github.com/DevSinghSachan/emdr2)          |                |
| [RETRO + DPR](https://arxiv.org/pdf/2112.04426v3.pdf)  | $45.5$ |                                                              |                |
| [RAG](https://arxiv.org/pdf/2005.11401v4.pdf)          | $44.5$ |                                                              |                |
| [DPR](https://arxiv.org/pdf/2004.04906v3.pdf)          | $41.5$ | [Official](https://github.com/facebookresearch/DPR)          |                |
| [REALM](https://arxiv.org/pdf/2002.08909v1.pdf)        | $40.4$ | [Official](https://github.com/google-research/language/tree/master/language/realm) |                |
| [PaLM-540B](https://arxiv.org/pdf/2204.02311v3.pdf)    | $39.6$ | [Official](https://github.com/lucidrains/PaLM-pytorch)       |                |
| [Chinchilla](https://arxiv.org/pdf/2203.15556v1.pdf)   | $35.5$ |                                                              |                |
| [GLaM 62B/64E](https://arxiv.org/pdf/2112.06905v2.pdf) | $32.5$ |                                                              |                |
| [GPT-3 175B](https://arxiv.org/pdf/2005.14165v4.pdf)   | $29.9$ | [Official](https://github.com/openai/gpt-3)                  |                |
| [Gopher](https://arxiv.org/pdf/2112.11446v2.pdf)       | $28.2$ |                                                              |                |
| [Neo-6B](https://arxiv.org/pdf/2210.02441v2.pdf)       | $19.7$ | [Official](https://github.com/hazyresearch/ama_prompting)    |                |

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

