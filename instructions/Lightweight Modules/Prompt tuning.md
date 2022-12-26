## Prompt tuning

You can use Prompt tuning (though only for `BART`, `T5`, and `GPT-2`) module through setting ``efficient_methods`` and ``efficient_kwargs``. 

For example, you can use Prompt tuning in BART following below commands:

```bash
python run_textbox.py \
    --model=BART \
    --model_path=facebook/bart-base \
    --dataset=samsum \
    --efficient_methods=\[\'prompt-tuning\'\] \
    --efficient_kwargs=\{\'prompt_length\':\ 100\}
```
or yaml equivalent:
```yaml
efficient_methods: ['prompt-tuning']
efficient_kwargs: {'prompt_length': 100}
```