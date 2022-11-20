## Pegasus

You can fine-tune a Pegasus model from HuggingFace through ``model=Pegasus``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=Pegasus \
    --model_path=google/pegasus-large \
    --dataset=samsum
```
