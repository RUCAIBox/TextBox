# -*- coding: utf-8 -*-
# @Time   : 2020/8/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.utils.enum_type
#######################
"""

from enum import Enum


class ModelType(Enum):
    """Type of models.

    - ``UNCONDITIONAL``: Unconditional Generator
    - ``CONDITIONAL``: Conditional Generator
    """

    UNCONDITIONAL = 1
    CONDITIONAL = 2
    SEQUENTIAL = 3


class DataLoaderType(Enum):
    """Type of DataLoaders.

    - ``ORIGIN``: Original DataLoader
    - ``FULL``: DataLoader for full-sort evaluation
    - ``NEGSAMPLE``: DataLoader for negative sample evaluation
    """

    UNCONDITIONAL = 1
    CONDITIONAL = 2


class EvaluatorType(Enum):
    """Type for evaluation metrics.

    - ``RANKING``: Ranking metrics like NDCG, Recall, etc.
    - ``INDIVIDUAL``: Individual metrics like AUC, etc.
    """

    RANKING = 1
    INDIVIDUAL = 2


class InputType(Enum):
    """Type of Models' input.

    - ``POINTWISE``: Point-wise input, like ``uid, iid, label``.
    - ``PAIRWISE``: Pair-wise input, like ``uid, pos_iid, neg_iid``.
    """

    NOISE = 1
    TEXT = 2


class FeatureType(Enum):
    """Type of features.

    - ``TOKEN``: Token features like user_id and item_id.
    - ``FLOAT``: Float features like rating and timestamp.
    - ``TOKEN_SEQ``: Token sequence features like review.
    - ``FLOAT_SEQ``: Float sequence features like pretrained vector.
    """

    TOKEN = 'token'
    FLOAT = 'float'
    TOKEN_SEQ = 'token_seq'
    FLOAT_SEQ = 'float_seq'


class FeatureSource(Enum):
    """Source of features.

    - ``CORPUS``: Features from ``.corpus`` (other than ``word_id``).
    - ``USER``: Features from ``.word`` (other than ``word_id``).
    - ``WORD_ID``: ``word_id`` feature.
    """

    CORPUS = 'corpus'
    WORD = 'word'
    WORD_ID = 'word_id'


class SpecialTokens:
    r"""Special tokens, including :attr:`PAD`, :attr:`UNK`, :attr:`BOS`, :attr:`EOS`.
    These tokens will by default have token ids 0, 1, 2, 3,
    respectively.
    """
    PAD = "<PAD>"
    UNK = "<UNK>"
    SOS = "<SOS>"
    EOS = "<EOS>"
