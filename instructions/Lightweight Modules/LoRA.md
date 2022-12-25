## LoRA

You can use LoRA module (though only for `BART`, `T5`, and `GPT-2`) through setting ``efficient_methods`` and ``efficient_kwargs``. 

For example, you can use LoRA in BART following below commands:

```bash
python run_textbox.py \
    --model=BART \
    --model_path=facebook/bart-base \
    --dataset=samsum \
    --efficient_methods=\[\'lora\'\] \
    --efficient_kwargs=\{\'lora_r\':\ 4,\ \'lora_dropout\':\ 0.1,\ \'lora_alpha\':\ 32\}
```
or yaml equivalent:
```yaml
efficient_methods: ['lora']
efficient_kwargs: {'lora_r': 4, 'lora_dropout': 0.1, 'lora_alpha': 32}
```