## Chinese-T5

You can fine-tune a Chinese-T5 model from HuggingFace through ``model=T5``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=T5 \
    --model_path=Langboat/mengzi-t5-base \
    --dataset=csl
```
