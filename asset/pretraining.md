# Pre-training

Pre-training models from scratch or continue pre-training from existing checkpoints is essential to achieving SOTA results. TextBox 2.0 provides four pre-training objectives to help pre-train a model from scratch:

- language modeling
- masked sequence-to-sequence modeling
- denoising auto-encoding 
- masked span prediction

To enable pre-training, simply set `--pretrain_task` (by default, pre-training is disabled and thus set to `disabled`), `--dataset`, and `--model` to specify the pre-training task, pre-training corpus, and architecture. 

The following example shows an example that pre-trains a Chinese BART on the WuDaoCorpora using the denoising pre-training objective from scratch. Our code will select a checkpoint with the best NLL loss on the validation set.

```bash
python run_textbox.py --model=transformer --dataset=wudao --pretrain_task=denoising
```

The WuDaoCorpora can be found at the link: https://resource.wudaoai.cn/home. Then, you should split it into a training set and a validation set. You only need to create two files, `train.src` and `valid.src`, since pre-training is an unsupervised process (no need for a target file) and does not need the test process (no need for a test file). This instruction is also suitable for your own pre-training corpus.

Moreover, the above script will pre-train a randomly initialized transformer, which has the same size as `bart-base`. If you want to modify the size, please refer to the instructions for the [transformer](https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/transformer.md).

If you want to continue pre-training a model, such as `bart-large`, on your dataset `xxx`, you can change the `--model` and `--model_path` options.

```bash
python run_textbox.py --model=BART --model_path=facebook/bart-large --dataset=xxx --pretrain_task=denoising
```
