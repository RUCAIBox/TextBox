# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

"""
recbole.utils.enum_type
#######################
"""

from enum import Enum


class ModelType(Enum):
    """Type of models.

    - ``UNCONDITIONAL``: Unconditional Generator
    - ``TRANSLATION``: Translation Model
    """

    UNCONDITIONAL = 1
    CONDITIONAL = 2
    TRANSLATION = 3


class DataLoaderType(Enum):
    """Type of DataLoaders.

    - ``UNCONDITIONAL``: Unconditional DataLoader
    - ``TRANSLATION``: DataLoader for translation dataset
    """

    UNCONDITIONAL = 1
    TRANSLATION = 2


class EvaluatorType(Enum):
    """Type for evaluation metrics.

    - ``RANKING``: Ranking metrics like NDCG, Recall, etc.
    - ``INDIVIDUAL``: Individual metrics like AUC, etc.
    """

    RANKING = 1
    INDIVIDUAL = 2


class InputType(Enum):
    """Type of Models' input.

    - ``NOISE``: Noise input.
    - ``PAIRTEXT``: Pair-wise input, like ``src_text, tar_text``.
    """

    NOISE = 1
    PAIRTEXT = 2


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
    PAD = "<|pad|>"
    UNK = "<|unk|>"
    SOS = "<|startoftext|>"
    EOS = "<|endoftext|>"
