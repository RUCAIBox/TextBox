## LongT5

You can fine-tune a longt5 model from HuggingFace through ``model=longt5``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=longt5 \
    --model_path=google/long-t5-local-base \
    --dataset=samsum
```
