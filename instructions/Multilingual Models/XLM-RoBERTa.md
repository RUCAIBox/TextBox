## XLM-RoBERTa

You can fine-tune a XLM-RoBERTa model from HuggingFace through ``model=XLM-RoberTa``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. Specifically, XLM-RoBERTa is used in Encoder-Decoder Framework for text generation.

Example usage:

```bash
python run_textbox.py \
    --model=XLM-RoBERTa \
    --model_path=xlm-roberta-base \
    --dataset=wmt16-ro-en 
```
