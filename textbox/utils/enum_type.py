class SpecialTokens:
    r"""Special tokens, including :attr:`PAD`, :attr:`UNK`, :attr:`BOS`, :attr:`EOS`.
    These tokens will by default have token ids 0, 1, 2, 3,
    respectively.
    """
    PAD = "[PAD]"
    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"


PLM_MODELS = [
    # CLM_MODELS
    'gpt2', 'gpt', 'big_bird', 'bert', 'roberta', 'cpm', 'ctrl', 'megatron_bert', 'transfo_xl', 'gpt_neo',
    # EncDecLM_MODELS
    't5', 'mt5', 'bart', 'led', 'mbart', 'bert2bert', 'big_bird_pegasus', 'pegasus', 'blender_bot', 'blender_bot_small',
    'm2m100', 'prophet_net'
]