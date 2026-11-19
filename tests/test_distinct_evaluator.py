from importlib import import_module
from itertools import chain
import math
from pathlib import Path
from statistics import mean
import sys
from types import ModuleType, SimpleNamespace
import warnings

from nltk.tokenize import word_tokenize
import pytest


TEXT = ["alpha alpha beta", "gamma delta gamma delta"]
TOKENS = [text.split() for text in TEXT]


@pytest.fixture(scope="module")
def evaluator_modules():
    # Import the real evaluator package without loading TextBox's model stack.
    package = ModuleType("_textbox_distinct_test")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "textbox")]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, package.__name__, package)
        try:
            base = import_module(package.__name__ + ".evaluator.base_evaluator")
            distinct = import_module(package.__name__ + ".evaluator.distinct_evaluator")
            yield base, distinct.DistinctEvaluator
        finally:
            for name in tuple(sys.modules):
                if name.startswith(package.__name__ + "."):
                    sys.modules.pop(name)


def expected_scores(tokens, max_order, *, inter, avg):
    def score(ngrams):
        return (len(set(ngrams)) + 1e-12) / (len(ngrams) + 1e-5) * 100

    expected = {}
    for order in range(1, max_order + 1):
        per_sentence = [
            [tuple(row[start:start + order]) for start in range(len(row) - order + 1)]
            for row in tokens
        ]
        if inter:
            value = score(list(chain.from_iterable(per_sentence)))
            expected[f"distinct-{order}"] = round(value, 2) if avg else value
        else:
            values = [score(ngrams) for ngrams in per_sentence]
            expected[f"distinct-{order}"] = round(mean(values), 2) if avg else values
    return expected


def evaluate(evaluator_modules, tokens, *, inter, avg, max_order=2):
    _, evaluator = evaluator_modules
    config = {"inter_distinct": inter, "distinct_max_ngrams": max_order}
    return evaluator(config).evaluate(SimpleNamespace(tokens=tokens), None, avg=avg)


def assert_scores(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert isinstance(actual[key], list) == isinstance(value, list)
        assert actual[key] == pytest.approx(value, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("avg", [True, False])
def test_corpus_scores(evaluator_modules, avg):
    actual = evaluate(evaluator_modules, TOKENS, inter=True, avg=avg)
    assert_scores(actual, expected_scores(TOKENS, 2, inter=True, avg=avg))
    if avg:
        assert actual == {"distinct-1": 57.14, "distinct-2": 80.0}


@pytest.mark.parametrize("avg", [True, False])
def test_sentence_scores(evaluator_modules, avg):
    actual = evaluate(evaluator_modules, TOKENS, inter=False, avg=avg)
    assert_scores(actual, expected_scores(TOKENS, 2, inter=False, avg=avg))
    if avg:
        assert actual == {"distinct-1": 58.33, "distinct-2": 83.33}


@pytest.mark.parametrize("inter", [True, False])
def test_single_sentence(evaluator_modules, inter):
    tokens = [TOKENS[0]]
    actual = evaluate(evaluator_modules, tokens, inter=inter, avg=False)
    assert_scores(actual, expected_scores(tokens, 2, inter=inter, avg=False))


def test_higher_orders_and_short_sentences(evaluator_modules):
    tokens = [["alpha"], *TOKENS]
    actual = evaluate(evaluator_modules, tokens, inter=False, avg=False, max_order=5)
    assert_scores(actual, expected_scores(tokens, 5, inter=False, avg=False))
    assert actual["distinct-5"] == pytest.approx([1e-5] * len(tokens))


@pytest.mark.parametrize("avg", [True, False])
def test_empty_corpus_keeps_corpus_smoothing(evaluator_modules, avg):
    actual = evaluate(evaluator_modules, [], inter=True, avg=avg)
    assert_scores(actual, expected_scores([], 2, inter=True, avg=avg))


@pytest.mark.parametrize("avg", [True, False])
def test_empty_corpus_inherits_list_aggregation(evaluator_modules, avg):
    # AbstractEvaluator's mean over an empty list is undefined.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = evaluate(evaluator_modules, [], inter=False, avg=avg)
    assert actual.keys() == {"distinct-1", "distinct-2"}
    if avg:
        assert all(math.isnan(value) for value in actual.values())
    else:
        assert actual == {"distinct-1": [], "distinct-2": []}


@pytest.mark.parametrize("avg", [True, False])
def test_base_evaluator_dispatch(evaluator_modules, monkeypatch, avg):
    base, _ = evaluator_modules
    # These one-line fixtures need word tokenization only, with no Punkt data.
    monkeypatch.setattr(base, "word_tokenize", lambda text: word_tokenize(text, preserve_line=True))
    config = {
        "lower_evaluation": False,
        "remove_punc": False,
        "is_chinese_task": False,
        "distinct_max_ngrams": 2,
        "inter_distinct": False,
    }
    evaluator = base.BaseEvaluator(config, ["distinct"])
    reference = SimpleNamespace(target_text=["alpha", "delta"])
    kwargs = {} if avg else {"avg": False}
    assert_scores(evaluator.evaluate(TEXT, reference, **kwargs),
                  expected_scores(TOKENS, 2, inter=False, avg=avg))
