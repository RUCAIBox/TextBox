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
    """

    UNCONDITIONAL = 1
    GAN = 2
    SEQ2SEQ = 3
    ATTRIBUTE = 4


class SpecialTokens:
    r"""Special tokens, including :attr:`PAD`, :attr:`UNK`, :attr:`BOS`, :attr:`EOS`.
    These tokens will by default have token ids 0, 1, 2, 3,
    respectively.
    """
    PAD = "<|pad|>"
    UNK = "<|unk|>"
    SOS = "<|startoftext|>"
    EOS = "<|endoftext|>"


PLM_MODELS = [
    # CLM_MODELS
    'gpt2', 'gpt', 'big_bird', 'bert', 'roberta', 'cpm', 'ctrl', 'megatron_bert', 'transfo_xl', 'gpt_neo',
    # EncDecLM_MODELS
    't5', 'mt5', 'bart', 'led', 'mbart', 'bert2bert', 'big_bird_pegasus', 'pegasus', 'blender_bot', 'blender_bot_small',
    'm2m100', 'prophet_net'
]