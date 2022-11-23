## CPM

You can fine-tune a CPM model from HuggingFace through ``model=CPM``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=CPM \
    --model_path=TsinghuaAI/CPM-Generate \
    --dataset=csl
```
