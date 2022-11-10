## BERT2BERT

BERT2BERT uses the Encoder-Decoder framework and encoder and decoder are initialized by the pretrained model. You can fine-tune a BERT2NERT model from through ``model=BERT2BERT``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=BERT2BERT \
    --model_path=bert-base-uncased \
    --dataset=samsum
```
