## Chinese-GPT2

You can fine-tune a Chinese-GPT2 model from HuggingFace through ``model=Chinese-GPT2``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=Chinese-GPT2 \
    --model_path=ckiplab/gpt2-base-chinese \
    --dataset=csl
```
