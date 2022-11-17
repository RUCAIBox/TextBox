## CPT

You can fine-tune a CPT model from HuggingFace through ``model=CPT``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py \
    --model=CPT \
    --model_path=fnlp/cpt-large \
    --dataset=csl
```
