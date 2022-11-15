## DistilBART

You can fine-tune a DistilBART model from HuggingFace through ``model=BART``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=BART \
    --model_path=sshleifer/distilbart-cnn-12-6 \
    --dataset=samsum
```
