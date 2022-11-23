## CRTL
You can fine-tune a CTRL model from HuggingFace through ``model=CTRL``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=CTRL \
    --model_path=ctrl \
    --dataset=samsum
```
