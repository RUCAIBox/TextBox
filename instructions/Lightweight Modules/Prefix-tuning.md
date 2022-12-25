## Prefix-tuning

You can use Prefix-tuning module (though only for `BART`, `T5`, and `GPT-2`) through setting ``efficient_methods`` and ``efficient_kwargs``. 

For example, you can use Prefix-tuning in BART following below commands:

```bash
python run_textbox.py \
    --model=BART \
    --model_path=facebook/bart-base \
    --dataset=samsum \
    --efficient_methods=\[\'prefix-tuning\'\] \
    --efficient_kwargs=\{\'prefix_length\':\ 100,\ \'prefix_dropout\':\ 0.1,\ \'prefix_mid_dim\':\ 512\}
```
or yaml equivalent:
```yaml
efficient_methods: ['prefix-tuning']
efficient_kwargs: {'prefix_length': 100, 'prefix_dropout': 0.1, 'prefix_mid_dim': 512}
```