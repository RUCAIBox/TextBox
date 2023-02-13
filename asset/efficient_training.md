# Efficient Training

### Distributed Data Parallel

TextBox supports training models with multiple GPUs and FP16 method based on `accelerate`. Configure and test (not always necessary) [accelerate](https://github.com/huggingface/accelerate) with `accelerate config` and `accelerate test` in the shell ([example](https://github.com/RUCAIBox/TextBox/blob/2.0.0/asset/accelerate.md)).

Once running `accelerate config`, you can run the code in the same configuration without setting the config again. If you want to change the configuration of `accelerate`, just re-run `accelerate config` to reset the configuration. If you don't want to use `accelerate`, run the code using `python` command.

To accelerate multi-GPU training with `accelerate`, run the script below. Note that the main process of `accelerate` listens to port `main_process_port`. Please manually set it to a different port if you run multiple TextBox instances on a single machine with FSDP (Fully Sharded Data Parallel) enabled.

```bash
accelerate launch [--main_process_port <port-number>] run_textbox.py ...
```

In multi-GPU training, you should set the hyper-parameter `gpu_id` to decide the devices in training. And the number of GPUs set in `gpu_id` is necessarily greater or equal to the number of GPUs set in `accelerate`.

```bash
accelerate launch [--main_process_port <port-number>] \
        run_textbox.py ... --gpu_id=<gpu-ids>
```

Note that `gpu_ids` is the usable GPU id list (such as `0,1,2,3`).

If you face an issue about `find_unused_parameters`:
```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
making sure all `forward` function outputs participate in calculating loss.
```
You should follow the advice to pass `find_unused_parameters=True` in the command line:
```bash
accelerate launch [--main_process_port <port-number>] \
        run_textbox.py ... --gpu_id=<gpu-ids> --find_unused_parameters=True
```

### Efficient Decoding

To further accelerate the decoding efficiency, we integrate FastSeq to optimize the decoding process by attention cache optimization, repeated n-gram detection, and asynchronous parallel I/O.

### Hyper-Parameters Tuning

A separate script `run_hyper.py` is provided for hyper-parameters tuning. Use `space=<path-to-space-file>` and `algo=<algo-name>` to select from different configurations.

#### Configuration of Search Space

Create a text file and define parameters to be tuned line by line in the format below:

```text
<label> <algorithm> <space>
```

- `<label>`: Parameters to be optimized
- `<algorithm>`: Algorithm to define search space, like `choice` and `loguniform`.
- `<space>`: Search space. Any Python object within one line is supported.

Search space should be defined accordingly to the algorithm:

- `choice`: `<space>` receives **an iterable of choices**.
- `loguniform`: `<space>` receives **an iterable of positional arguments (low, high)**, which returns a value drawn according to $exp(uniform(low, high))$.
- more algorithm visit [parameter expressions](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/) for a full support list.

#### Example

```text
# hyperopt_example.test
learning_rate loguniform (-8, 0)
embedding_size choice [64, 96 , 128]
train_batch_size choice [512, 1024, 2048]
```

```bash
# command line instruction
python run_hyper.py --space=textbox/properties/hyperopt_example.test --algo='exhaustive'  --model_path=facebook/bart-base --metrics=\[\'rouge\'\] --metrics_for_best_model=\[\'ROUGE-1\'\]
```

### Multiple Random Seeds

Similar to hyper-parameters tuning, another python code `run_multi_seed.py` with new parameter `multi_seed=<int>` indicating the amount of seeds to be tested, is introduced for multiple random seeds test:

```bash
python run_multi_seed.py --multi_seed=16  --model_path=facebook/bart-base --metrics=\[\'rouge\''\] --metrics_for_best_model=\[\'ROUGE-1\'\]
```

Specify `seed` parameter to reproduce the generation of multiple seeds.