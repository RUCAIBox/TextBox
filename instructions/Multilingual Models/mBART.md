## mBART

You can fine-tune a mBART model from HuggingFace through ``model=mBART``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. Specifically, mBART is a Multilingual Model. You can set source language and target language with ``src_lang=<source language>``, ``tgt_lang=<target_language>``.

Example usage:

```bash
python run_textbox.py \
    --model=mBART \
    --model_path=facebook/mbart-large-cc25 \
    --dataset=wmt16-ro-en \
    --src_lang=ro_RO \
    --tgt_lang=en_XX
```
