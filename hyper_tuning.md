
### Auto-tuning Hyperparameter 

#### Configuration

Open `TextBox/hyper.test` and set several hyperparameters to auto-searching in parameter list. 

```text
<param> <method> <space>
```

- `<param>`: Parameters to be optimized
- `<method>`: Supports following method 
    - **choice**: indicates that the parameter takes discrete values from the setting list.
    - **uniform**:
    - **quniform**:
    - **loguniform**: indicates that the parameters obey the uniform distribution, randomly taking values from $e^{-8}$ to $e^{0}$.
- `<space>`: Search space. Any Python object within one line is supported.

Here is an example for `hyper.test`: 

```text
learning_rate loguniform -8, 0
embedding_size choice [64, 96 , 128]
train_batch_size choice [512, 1024, 2048]
mlp_hidden_size choice ['[64, 64, 64]','[128, 128]']
```

#### Run

Set training command parameters as you need to run:

```bash
python run_hyper.py --model=<model-name> --dataset=<data-name> --config_files=<yaml-config> --params_file=<hyper-test>
```

e.g.

```bash
python run_hyper.py --model=BPR --dataset=ml-100k --config_files=test.yaml --params_file=hyper.test
```

Note that `--config_files=test.yaml` is optional, if you don't have any customize config settings, this parameter can be empty.

This processing maybe take a long time to output best hyperparameter and result:
```bash
running parameters:                                                                                                                    
{'embedding_size': 64, 'learning_rate': 0.005947474154838498, 'mlp_hidden_size': '[64,64,64]', 'train_batch_size': 512}                
  0%|                                                                                           | 0/18 [00:00<?, ?trial/s, best loss=?]
```

