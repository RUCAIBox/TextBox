class SpecialTokens:
    r"""Special tokens, including :attr:`PAD`, :attr:`UNK`, :attr:`BOS`, :attr:`EOS`.
    These tokens will by default have token ids 0, 1, 2, 3,
    respectively.
    """
    PAD = "[PAD]"
    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"

CLM_MODELS = ["cpm", "ctrl", "gpt2", "gpt_neo", "openai-gpt", "opt"]
SEQ2SEQ_MODELS = ["bart", "bert2bert", "bigbird_pegasus", "blenderbot", "blenderbot-small", "led", "m2m_100", "mbart", "mt5", "mvp", "pegasus", "prophetnet", "t5"]

PLM_MODELS = CLM_MODELS + SEQ2SEQ_MODELS