## Adapter

You can use Adapter module (though only for `BART`, `T5`, and `GPT-2`) through setting ``efficient_methods`` and ``efficient_kwargs``. 

For example, you can use Adapter in BART following below commands:

```bash
python run_textbox.py \
    --model=BART \
    --model_path=facebook/bart-base \
    --dataset=samsum \
    --efficient_methods=\[\'adapter\'\] \
    --efficient_kwargs=\{\'adapter_mid_dim\':\ 64\}
```
or yaml equivalent:
```yaml
efficient_methods: ['adapter']
efficient_kwargs: {'adapter_mid_dim': 64}
```