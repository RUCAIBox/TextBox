# TextBox 1.0.0    Go! Go! Go!

## Overall
- [ ] Perfectly reproduce and recover, save and load random seed state. Work with dataloader and trainer checkpoint
- [ ] Resume training? Loading existing parameters? (how to judge?) Work with trainer checkpoint
- [ ] Only generation and only evaluation. How to solve?
- [ ] Quick test the whole pipeline (and `max_length`). Work with dataloader.
- [ ] Logger with DDP
- [ ] Reminder through email
- [ ] Hyper-parameter tuning (e.g. batch-size 16, 32, 64) (https://github.com/RUCAIBox/RecBole/blob/master/run_hyper.py, https://recbole.io/docs/user_guide/usage/parameter_tuning.html) (without saving model)
- [ ] Run on several random seeds and average their results (without saving model)
- [ ] Model deployment (https://clip-as-service.jina.ai/)
- [ ] Check `print` and `logger.info`
- [ ] Check `get_model`, `get_trainer`, `get_dataset` and `get_dataloader`
- **Simplfy import relation, add useful module in `__init__.py`** (for example `PLM_MODELS`)
- **Do not use `import *`**

## Config
- [ ] Argparse? config?
- [ ] Config check? (How to deal with wrong config?)
- [ ] Print all the config and command line (is `argument_list.py` necessary?) (maybe 3 classes `general`, `model` and `dataset`)
- [ ] Simplify `init_logger`, remove `is_logger` and support user-defined filename
- [ ] Case of model class, model file and model yaml (same for dataset)

## Dataset / Dataloader
### Overall
- [ ] Use `dataset` and `dataloader` from PyTorch, only with `source_id`, `source_text`, `target_id`, `target_text`. Does test set need dataloader? (just source side?)
- [ ] Load tokenizer and tokenize text here
- [ ] Download and process dataset automatically
- [ ] Save processed files. How to check is config or file changed?
- [ ] *Max-token?
- [ ] `eval` and `repr`
### Pre-training tasks (construct new `target_id` and `source_id` according to old `source_id`)
- [ ] DAE (like BART)
- [ ] Masked Seq2Seq (like MASS)
- [ ] Masked span prediction (like T5)
- [ ] LM (like GPT-2) Decoder? Encoder-decoder?
- [ ] *PLM (like MPNet) one-tower?
### *Others
- [ ] Support weighted sampling (change `sampler`?) (Note randomness!! https://pytorch.org/docs/stable/notes/randomness#dataloader)
- [ ] Support local reading (especially for pre-training, change `collate_fn`?)

## Trainer
### Efficiency
- [ ] Multi-GPU with `accelerate`? Need to research! (PyTorch or accelerate)
    - [ ] check `find_unused_parameters`
```
data parallel (model can fit in one GPU)
single node or several nodes
e.g. fine-tune or pre-train BART-large on a large dataset (16GB)
can save and load model and optimizer correctly
print log only once

see HF how to solve it:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py)
https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization_no_trainer.py
```
- [ ] Fast generation (with https://github.com/microsoft/fastseq)
- [ ] Multi-GPU generation (divide data to multiple GPUs for generation.) Is it possible? Under DDP?
- [ ] FP16 (maybe with HF?)
### Useful features
- [ ] Tensorboard to record loss and metric
- [ ] Support train and valid for several steps
- [ ] Support generation and evaluation during generation
### Save checkpoint
- [ ] Checkpoint format? (following HF?)
```
        model parameters (all) (to `cpu`)
        optimizer (trained)
        random state
        config
        valid results
```
- [ ] Save checkpoint and generated text every validation
### Others
- [ ] Check `_check_metrics`
- [ ] Add optimizer `AdamW` and `Adafactor` (for T5)
- [ ] Hyper-parameter for optimizer
- [ ] Only pass tuned parameters (`requires_grad=True`) to optimizer
- [ ] Simplify useless code
- [ ] `tqdm` with loss (metric) and `dynamic_ncols=True`
- [ ] Check `torch.no_grad()`, `model.train()` and `model.eval()`
- [ ] Move `optimizer` to `trainer` and change name to `scheduler`

## Model
### Overall
- [ ] Automaticly detect model name (case-insensitive)
- [ ] Simplify code using AutoModel and AutoTokenizer, following HF example
- [ ] Model without pre-trained weights
- [ ] Support `model_name`, `--tokenizer_name` and `--config_name`
- [ ] Check `__getattr__` in `abstract_model.py`
### Models in HF
- [ ] Add OPT
### Models not in HF
- [ ] Add UniLM (reproduce results on SQuAD)
- [ ] Add MASS (reproduce results on SQuAD)
- [ ] Add CPT and chinese BART (Add one Chinese dataset and reproduce results)
### Translation
- [ ] Add XLM (Add one translation dataset and reproduce results)
- [ ] Add MarianMT (the same)
- [ ] Test mBART, mT5 using the tranlastion dataset
### Non-pretrained models
- [ ] Refactor RNN Seq2Seq (merge `Attention`)
- [ ] Refactor Copied Seq2Seq
- [ ] Add basic Transformer
- [ ] Model initilazation (for PLM?)
### Prompting
- [ ] Add prompt tuning
- [ ] Add prefix tuning for GPT-2, BART, T5
- [ ] Add P-tuningv2 for GPT-2, BART, T5
- [ ] Add adapter for GPT-2, BART, T5
- [ ] Add LoRA for GPT-2, BART, T5
- [ ] Add bias tuning for GPT-2, BART, T5
### *Other models
- [ ] Add CTRL
- [ ] Add PPLM
- [ ] Add non-autoregressive models


## Evaluator
- [ ] Unify `base_evaluator`
- [ ] Support evaluation for different datasets and task. (how to specify the evaluation method?)
    - [ ] Text summarization: CNN/Daily Mail (cnndm), XSum (xsum), SAMSum (samsum), and WLE (wle).
    - [ ] Open-ended dialogue system: PersonaChat (pc), DailyDialog (dd), DSTC7-AVSD (da), and SGD (sgd).
    - [ ] Data-to-text generation: WebNLG v2.1 (webnlg), WebNLG v3.0 (webnlg2), WikiBio (wikibio), E2E (e2e), DART (dart), and ToTTo (totto).
    - [ ] Question generation: SQuAD (squadqg) and CoQA (coqaqg).
    - [ ] Story generation: ROCStories (roc) and WritingPrompts (wp).
    - [ ] Question answering: SQuAD (squad) and CoQA (coqa).
    - [ ] Task-oriented dialogue system: MultiWOZ 2.0 (multiwoz).
    - [ ] Commonsense generation: CommonGen (cg).
    - [ ] Text simplification: WikiAuto + Turk/ASSET (wia).
    - [ ] Paraphrase generation: Quora (comming soon).
    - [ ] Text style transfer: GYAFC-E&M and F&R (comming soon).


## Leaderboard
- [ ] Construct leaderboard for each datasets at GitHub page, including common models, paper link, metric results, and generated files (theirs (official link) or our reproduced (provide config)).
    - [ ] Text summarization: CNN/Daily Mail (cnndm), XSum (xsum), SAMSum (samsum), and WLE (wle).
    - [ ] Open-ended dialogue system: PersonaChat (pc), DailyDialog (dd), DSTC7-AVSD (da), and SGD (sgd).
    - [ ] Data-to-text generation: WebNLG v2.1 (webnlg), WebNLG v3.0 (webnlg2), WikiBio (wikibio), E2E (e2e), DART (dart), and ToTTo (totto).
    - [ ] Question generation: SQuAD (squadqg) and CoQA (coqaqg).
    - [ ] Story generation: ROCStories (roc) and WritingPrompts (wp).
    - [ ] Question answering: SQuAD (squad) and CoQA (coqa).
    - [ ] Task-oriented dialogue system: MultiWOZ 2.0 (multiwoz).
    - [ ] Commonsense generation: CommonGen (cg).
    - [ ] Text simplification: WikiAuto + Turk/ASSET (wia).
    - [ ] Paraphrase generation: Quora (comming soon).
    - [ ] Text style transfer: GYAFC-E&M and F&R (comming soon).

## Data statistics
- [ ] Further research