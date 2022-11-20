## Chinese-Pegasus

You can fine-tune a Chinese-Pegasus model from HuggingFace through ``model=Chinese-Pegasus``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=Chinese-Pegasus \
    --model_path=IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese \
    --dataset=csl
```
