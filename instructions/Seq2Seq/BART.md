## BART

You can fine-tune a BART model from HuggingFace through ``model=BART``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=BART \
    --model_path=facebook/bart-base \
    --dataset=samsum
```
