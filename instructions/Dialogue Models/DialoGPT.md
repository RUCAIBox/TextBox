## DialoGPT

You can fine-tune a DialoGPT model from HuggingFace through ``model=GPT2``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=GPT2 \
    --model_path=microsoft/DialoGPT-small \
    --dataset=dd
```
