## mT5

You can fine-tune a mT5 model from HuggingFace through ``model=mT5``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``.

Example usage:

```bash
python run_textbox.py \
    --model=mt5 \
    --model_path=google/mt5-small 
```
