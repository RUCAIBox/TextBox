# Basic Training
## config
You may want to load your configurations in equivalent ways:
* cmd
* config files
* yaml

### cmd 
You may want to change configurations in the command line like ``--xx=yy``. ``xx`` is the name of the parameters and ``yy`` is the corresponding value. for example:

```bash
python run_textbox.py --model=BART --model_path=facebook/bart-base --epochs=1
```

It's suitable for **a few temporary** modifications with cmd like:
* ``model``
* ``model_path``
* ``dataset``
* ``epochs``
* ...

### config files

You can also modify configurations through the local files:
```bash
python run_textbox.py ... --config_files <config-file-one> <config-file-two>
```

Every config file is an additional yaml file like:

```yaml
efficient_methods: ['prompt-tuning']
```
It's suitable for **a large number of** modifications or **long-term** modifications with cmd like:
* ``efficient_methods``
* ``efficient_kwargs``
* ...

### yaml 

The original configurations are in the yaml files. You can check the values there, but it's not recommended to modify the files except for **permanent** modification of the dataset. These files are in the path ``textbox\properties``:
* ``overall.yaml``
* ``dataset\*.yaml``
* ``model\*yaml``


## trainer

You can choose an optimizer and scheduler through `optimizer=<optimizer-name>` and `scheduler=<scheduler-name>`. We provide a wrapper around **pytorch optimizer**, which means parameters like `epsilon` or `warmup_steps` can be specified with keyword dictionaries `optimizer_kwargs={'epsilon': ... }` and `scheduler_kwargs={'warmup_steps': ... }`. See [pytorch optimizer](https://pytorch.org/docs/stable/optim.html#algorithms) and scheduler for a complete tutorial.  <!-- TODO -->

Validation frequency is introduced to validate the model **at each specific batch-steps or epoch**. Specify `valid_strategy` (either `'step'` or `'epoch'`) and `valid_steps=<int>` to adjust the pace. Specifically, the traditional train-validate paradigm is a special case with `valid_strategy=epoch` and `valid_steps=1`.

`max_save=<int>` indicates **the maximal amount of saved files** (checkpoint and generated corpus during evaluation). `-1`: save every file, `0`: do not save any file, `1`: only save the file with the best score, and `n`: save both the best and the last $n−1$ files.

According to ``metrics_for_best_model``, the score of the current checkpoint will be calculated, and evaluation metrics specified with ``metrics``([full list](evaluation.md)) will be chosen. **Early stopping** can be configured with `stopping_steps=<int>` and score of every checkpoint. 


```bash
python run_textbox.py ... --stopping_steps=8 \\
  --metrics_for_best_model=\[\'rouge-1\', \'rouge-w\'\] \\
  --metrics=\[\'rouge\'\]
```

You can resume from a **previous checkpoint** through ``model_path=<checkpoint_path>``.When you want to restore **all trainer parameters** like optimizer and start_epoch, you can set ``resume_training=True``. Otherwise, only **model and tokenizer** will be loaded. The script below will resume training from checkpoint in the path ``saved/BART-samsum-2022-Dec-18_20-57-47/checkpoint_best``

```bash
python run_textbox --model_path=saved/BART-samsum-2022-Dec-18_20-57-47/checkpoint_best \\
--resume_training=True
```

Other commonly used parameters include `epochs=<int>` and `max_steps=<int>` (indicating maximum iteration of epochs and batch steps, if you set `max_steps`, `epochs` will be invalid), `learning_rate=<float>`, `train_batch_size=<int>`, `weight_decay=<bool>`, and `grad_clip=<bool>`.

### Partial Experiment

You can run the partial experiment with `do_train`, `do_valid`and `do_test`. You can test your pipeline and debug with `quick_test=<amount-of-data-to-load>` to load just a few examples. 

The following script loads the trained model from a local path and conducts generation and evaluation without training and evaluation.
```bash
python run_textbox.py --model_path=saved/BART-samsum-2022-Dec-18_20-57-47/checkpoint_best \\
--do_train=False --do_valid=False
```

## wandb

If you are running your code in jupyter environments, you may want to log in by simply setting an environment variable (your key may be stored in plain text):

```python
%env WANDB_API_KEY=<your-key>
```
Here you can set wandb with `wandb`.

If you are debugging your model, you may want to **disable W&B** with `--wandb=disabled`, and **none of the metrics** will be recorded. You can also disable **sync only** with `--wandb=offline` and enable it again with `--wandb=online` to upload to the cloud. Meanwhile, the parameter can be configured in the yaml file like:

```yaml
wandb: online
```

The local files can be uploaded by executing `wandb sync` in the command line.

After configuration, you can throttle wandb prompts by defining the environment variable `export WANDB_SILENT=false`. For more information, see [documentation](docs.wandb.ai).