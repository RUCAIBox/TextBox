## OPT

You can fine-tune a OPT model from HuggingFace through ``model=OPT``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=OPT \
    --model_path=facebook/opt-350m \
    --dataset=samsum
```
