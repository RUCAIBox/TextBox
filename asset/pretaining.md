# Pre-training

Pre-training models from scratch or continue pre-training from existing checkpoints is essential to achieving SOTA results. TextBox 2.0 provides four pre-training objectives to help pre-train a model from scratch:

- language modeling
- masked sequence-to-sequence modeling
- denoising auto-encoding 
- masked span prediction

To enable pre-training, simply set `--pretrain_task`(by default, pre-training is disabled and thus set to `disabled`), `--dataset`, and `model` to specify the pre-training task, pre-training corpus, and architecture. 

The following example shows an example that pre-trains a Chinese BART on the WuDaoCorpora using the denoising pre-training objective.

```bash
python run_textbox.py ... --pretrain_task=<task-name>
```
