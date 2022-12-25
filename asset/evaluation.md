# Evaluation

### Evaluation Metrics

To conduct evaluations, 17 evaluation metrics belonging to 4 categories are integrated into TextBox 2.0. Lexical metrics are used to measure the n-gram overlap between generated texts and golden texts. Semantic metrics are used to compare the texts at sentence level. Diversity metrics are used to evaluate the lexical diversity of generated texts. Accuracy metrics are used to calculate the precision of important phrases.

- Lexical metrics: bleu, chrf, chrf+, chrf++, cider, meteor, nist, rouge, spice, ter
- Semantic metrics: bert_score, style strength
- Diversity metrics: distinct, self_bleu, unique
- Accuracy metrics: em, f1, inform, success

> Warning Backslashes and no-extra-space are required when inputting a list of strings like `\[\'bleu\',\'rouge\'\]` in command line. As a result, a preset run configuration is more recommended.

### Evaluation Parameters

Evaluation metrics can be specified with `metrics` ([full list](###evaluation-metrics)), and produce a dictionary of results:

```
python run_textbox.py ... --metrics=\[\'rouge\'\]
# results: { 'rouge-1': xxx, 'rouge-2': xxx, 'rouge-l': xxx, 'rouge-w': xxx, ... }
```

After specifying several evaluation metrics, further configuration on them is as follows:

For example, `rouge` provides `rouge_max_ngrams` and `rouge_type` to specify the maximal number of n-grams and type of rouge (like `files2rouge`, `rouge-score`, etc.). In addition, `bleu` provides `bleu_max_ngrams`, `bleu_type`, `smoothing_function=<int>`, and `corpus_bleu=<bool>` to customize metric.

```
bleu_max_ngrams: 4
bleu_type: nltk
smoothing_function: 0
corpus_bleu: False

distinct_max_ngrams: 4
```

Other evaluation metrics observe the same naming rules.

### Advanced Analysis

Besides the analysis using automatic metrics, TextBox 2.0 provides several visualization tools to explore and analyze the generated texts in various dimensions. A separate script `run_analysis.py` is provided for advanced analysis.

```bash
python run_analysis.py --dataset=cnndm  BART_output.txt T5_output.txt
```