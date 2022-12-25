## Pegasus_X

You can fine-tune a Pegasus_X model from HuggingFace through ``model=pegasus_x``, ``model_path=<hf-or-local-path>``, ``dataset=<dataset-name>``. 

Example usage:

```bash
python run_textbox.py  \
	--model_path=google/pegasus-x-base  \
	--model=Pegasus_X  \
	--dataset=samsum
```