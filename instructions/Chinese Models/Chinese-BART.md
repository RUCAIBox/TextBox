## Chinese-BART

You can fine-tune a Chinese-BART model from HuggingFace through ``model=Chinese-BART``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=Chinese-BART \
    --model_path=fnlp/bart-base-chinese \
    --dataset=csl
```
