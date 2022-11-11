## BigBird-Pegasus

You can fine-tune a BigBird-Pegasus model from HuggingFace through ``model=BigBird_Pegasus``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=BigBird_Pegasus \
    --model_path=google/bigbird-pegasus-large-arxiv \
    --dataset=samsum
```