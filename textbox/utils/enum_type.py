# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn

# UPDATE:
# @Time   : 2020/11/15
# @Author : Tianyi Tang
# @Email  : steventang@ruc.edu.cn

"""
textbox.utils.enum_type
#######################
"""

from enum import Enum


class ModelType(Enum):
    """Type of models.

    - ``UNCONDITIONAL``: Unconditional Generator
    - ``GAN``: Generative Adversarial Net
    - ``SEQ2SEQ``: Seq2Seq Generator
    - ``ATTRIBUTE``: Attribute Generator
    - ``TwoStage``: TwoStage Generator
    """

    UNCONDITIONAL = 1
    GAN = 2
    SEQ2SEQ = 3
    ATTRIBUTE = 4
    TwoStage = 5


class SpecialTokens:
    r"""Special tokens, including :attr:`PAD`, :attr:`UNK`, :attr:`BOS`, :attr:`EOS`.
    These tokens will by default have token ids 0, 1, 2, 3,
    respectively.
    """
    PAD = "<|pad|>"
    UNK = "<|unk|>"
    SOS = "<|startoftext|>"
    EOS = "<|endoftext|>"
