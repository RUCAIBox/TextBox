## ProphetNet

You can fine-tune a ProphetNet model from HuggingFace through ``model=ProphetNet``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=ProphetNet \
    --model_path=microsoft/prophetnet-large-uncased \
    --dataset=samsum
```
