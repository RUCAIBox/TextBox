## XLM-ProphetNet

You can fine-tune a XLM-ProphetNet model from HuggingFace through ``model=XLM-ProphetNet``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=XLM-ProphetNet \
    --model_path=microsoft/prophetnet-large-uncased \
    --dataset=wmt16-ro-en \
    --src_lang=ro \
    --tgt_lang=en
```
