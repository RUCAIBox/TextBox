## OpenAI-GPT

You can fine-tune a OpenAI-GPT model from HuggingFace through ``model=OpenAI-GPT``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=OpenAI-GPT \
    --model_path=openai-gpt \
    --dataset=samsum
```
