## LongT5

You can fine-tune a longt5 model from HuggingFace through ``model=longt5``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=longt5 \
    --model_path=Stancld/longt5-tglobal-large-16384-pubmed-3k_steps \
    --dataset=samsum
```
