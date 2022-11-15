## XLM

You can fine-tune a XLM model from HuggingFace through ``model=XLM``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. Specifically, XLM-RoBERTa is used in Encoder-Decoder Framework for text generation. And XLM is a Multilingual Model. You can set source language and target language with ``src_lang=<source language>``, ``tgt_lang=<target_language>``.

Example usage:

```bash
python run_textbox.py \
    --model=XLM \
    --model_path=xlm-mlm-17-1280 \
    --dataset=wmt16-ro-en \
    --src_lang=ro \
    --tgt_lang=en
```
