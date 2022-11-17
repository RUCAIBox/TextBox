## MVP

You can fine-tune a MVP model from HuggingFace through ``model=MVP``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=MVP \
    --model_path=RUCAIBox/mvp \
    --dataset=samsum
```
