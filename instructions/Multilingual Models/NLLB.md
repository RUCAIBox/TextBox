## NLLB

You can fine-tune a NLLB model from HuggingFace through ``model=NLLB``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. Specifically, NLLB is a Multilingual Model. You can set source language and target language with ``src_lang=<source language>``, ``tgt_lang=<target_language>``.

Example usage:

```bash
python run_textbox.py \
    --model=NLLB \
    --model_path=facebook/nllb-200-distilled-600M \
    --dataset=wmt16-ro-en \
    --src_lang=ro \
    --tgt_lang=en
```
