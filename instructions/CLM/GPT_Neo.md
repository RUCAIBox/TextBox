## GPT_Neo

You can fine-tune a GPT_Neo model from HuggingFace through ``model=GPT_Neo``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=GPT_Neo \
    --model_path=EleutherAI/gpt-neo-125M \
    --dataset=samsum
```
