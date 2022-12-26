## BitFit

You can use BitFit module (though only for `BART`, `T5`, and `GPT-2`) through setting ``efficient_methods`` and ``efficient_kwargs``. 

For example, you can use BitFit in BART following below commands:

```bash
python run_textbox.py \
    --model=BART \
    --model_path=facebook/bart-base \
    --dataset=samsum \
    --efficient_methods=\[\'bitfit\'\] \
    --efficient_kwargs=\{\}
```
or yaml equivalent:
```yaml
efficient_methods: ['bitfit']
efficient_kwargs: {}
```