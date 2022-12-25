## MASS

You can fine-tune a MASS model from HuggingFace through ``model=MASS``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=MASS \
    --model_path=RUCAIBox/mass-base-uncased \
    --dataset=samsum
```
