## UniLM

You can fine-tune a T5 model from HuggingFace through ``model=T5``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=UniLM \
    --model_path=microsoft/unilm-base-cased \
    --dataset=samsum
```
