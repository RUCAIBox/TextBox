## Blenderbot-Small

You can fine-tune a Blenderbot-Small model from HuggingFace through ``model=Blenderbot-Small``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=Blenderbot-Small \
    --model_path=facebook/blenderbot_small-90M \
    --dataset=dd
```