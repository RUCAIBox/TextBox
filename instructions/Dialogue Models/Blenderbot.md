## Blenderbot

You can fine-tune a Blenderbot model from HuggingFace through ``model=Blenderbot``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=Blenderbot \
    --model_path=facebook/blenderbot-400M-distill \
    --dataset=dd
```
