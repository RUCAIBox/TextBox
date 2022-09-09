## Training from scratch
### Basics
You can train a transformer model from scratch through ``model=transformer``. The default parameters of the transformer are the same as ``bart-base``, and you can modify it in the ``textbox/property/model/transformer.yaml``

```bash
python run_textbox.py --model=transformer 
```

### Modify the model hyper-parameters:
You can also modify your transformer's hyper-parameters. For example, a transformer fewer layers can be configured:

```bash
python run_textbox.py --model=transformer --encoder_layers=2 --decoder_layers=2
```
or yaml equivalent:
```yaml
encoder_layers: 2
decoder_layers: 2
```