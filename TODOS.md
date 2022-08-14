# TextBox 2.0.0    Go! Go! Go!

## Overall
- [ ] Perfectly reproduce and recover, save and load random seed state. Work with dataloader and trainer checkpoint
- [ ] Resume training (default) Loading existing parameters? (add an options) Work with trainer checkpoint
- [x] Only generation and only evaluation. (`--do_train --do_test --do_eval`)
- [x] Quick test the whole pipeline (and `max_length`). Work with dataloader. (`--quick_test` with lazy load using `fseek`)
- [x] Logger with DDP
- [ ] Reminder through email (wandb)
- [x] Hyper-parameter tuning (e.g. batch-size 16, 32, 64) (https://github.com/RUCAIBox/RecBole/blob/master/run_hyper.py, https://recbole.io/docs/user_guide/usage/parameter_tuning.html) (without saving model)
- [x] Run on several random seeds and average their results (without saving model)
- [ ] Model deployment (https://clip-as-service.jina.ai/)
- [ ] Check `print` and `logger.info`
- [ ] Check `get_model`, `get_trainer`, `get_dataset` and `get_dataloader`
- [ ] Check `warnings.warn()` and `logger.warning()`
- **Simplfy import relation, add useful module in `__init__.py`** (for example `PLM_MODELS`)
- **Do not use `import *`**

## Config
- [x] Config check (user should add their own config in a file, eg `argument_list`)
- [x] Print all the config and command line (is `argument_list.py` necessary?) (maybe 3 classes `general`, `model` and `dataset`)
- [x] Simplify `init_logger`, remove `is_logger` and support user-defined filename
- [ ] Case of model class, model file and model yaml (same for dataset)

## Dataset / Dataloader
### Overall
- [x] Use `dataset` and `dataloader` from PyTorch, only with `source_id`, `source_text`, `target_id`, `target_text`. (optional `target_id`, `target_text`)
- [x] Load tokenizer and tokenize text here (support tokenize and token2idx seprately)
- [ ] Download and process dataset automatically
- [ ] Save processed files. How to check is config or file changed? (maybe with md5 of config and files)
- [ ] *Max-token?
- [ ] `eval` and `repr`
- [ ] valid target setting if not metric for best is not loss
- [ ] Add attribute `tokenized`?
### Pre-training tasks (construct new `target_id` and `source_id` according to old `source_id`)
- [x] DAE (like BART)
- [ ] Masked Seq2Seq (like MASS)
- [ ] Masked span prediction (like T5)
- [ ] LM (like GPT-2) Decoder? Encoder-decoder?
- [ ] *PLM (like MPNet) one-tower?
### *Others
- [ ] Support weighted sampling (change `sampler`?) (Note randomness!! https://pytorch.org/docs/stable/notes/randomness#dataloader)
- [ ] Support local reading (especially for pre-training, change `collate_fn`?)

## Trainer
### Efficiency
- [x] Multi-GPU with `accelerate`? Need to research! (PyTorch or accelerate)
    - [x] check `find_unused_parameters`
    - [x] will our scheduler be impacted
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
- [x] Fast generation (with https://github.com/microsoft/fastseq or https://github.com/bytedance/lightseq/tree/master/lightseq/training )
- [x] Multi-GPU generation (divide data to multiple GPUs for generation.) Is it possible? Under DDP?
- [x] *FP16 (HF? or Pytorch?)
### Useful features
- [x] WanDB to record loss and metric
- [x] Support train and valid for several steps
- [x] Support generation and evaluation during validation
### Save checkpoint
- [ ] Checkpoint format? (following HF?)
```
        model parameters (all) (to `cpu`)
        optimizer (trained)
        random state
        config
        valid results
```
- [x] Save checkpoint and generated text every validation
### Others
- [x] Check `_check_metrics`
- [x] Add optimizer `AdamW` and `Adafactor` (for T5)
- [x] Hyper-parameter for optimizer
- [x] Only pass tuned parameters (`requires_grad=True`) to optimizer
- [x] Simplify useless code
- [x] `tqdm` with loss (metric) and `dynamic_ncols=True`
- [x] Check `torch.no_grad()`, `model.train()` and `model.eval()`
- [x] Move `optimizer` to `trainer` and change name to `scheduler`

## Model
### Overall
- [ ] Automaticly detect model name (case-insensitive)
- [x] Simplify code using AutoModel and AutoTokenizer, following HF example
- [ ] Model without pre-trained weights
- [x] Support `model_name`, `--tokenizer_name` and `--config_name`
- [x] Check `__getattr__` in `abstract_model.py`
### Models in HF
- [x] Add OPT
### Models not in HF
- [ ] Add UniLM (reproduce results on SQuAD)
- [ ] Add MASS (reproduce results on SQuAD)
- [x] Add CPT and chinese BART (Add one Chinese dataset and reproduce results)
### Translation
- [ ] Add XLM (Add one translation dataset and reproduce results)
- [ ] Add MarianMT (the same)
- [x] Test mBART, mT5 using the tranlastion dataset
### Non-pretrained models
- [ ] Refactor RNN Seq2Seq (merge `Attention`)
- [ ] Refactor Copied Seq2Seq
- [ ] Add basic Transformer
- [ ] Model initilazation (for PLM?)
### Prompting
- [x] Add prompt tuning
- [x] Add prefix tuning for GPT-2, BART, T5
- [x] Add P-tuningv2 for GPT-2, BART, T5
- [x] Add adapter for GPT-2, BART, T5
- [x] Add LoRA for BART, T5
- [ ] Add LoRA, prompt tuning for GPT-2
- [x] Add bias tuning for GPT-2, BART, T5
- [ ] Right prompt tuning
### *Other models
- [x] Add CTRL
- [ ] Add PPLM
- [ ] Add non-autoregressive models


## Evaluator
- [x] Unify `base_evaluator`
- [ ] Refactor `files2rouge` and with `try` and `except`, and remove empty line
- [ ] `multi-bleu` traceback
- [ ] Add TED following https://github.com/PlusLabNLP/AESOP/blob/master/evaluation/eval.py
- [ ] Name and doc check
- [ ] Check `bert-score` HF logging
- [ ] Check and remake each dataset (especially, CoQA, webnlg)
- [ ] corpus.copy()
- [ ] Support evaluation for different datasets and task. (how to specify the evaluation method?)
    - [x] Text summarization: CNN/Daily Mail (cnndm), XSum (xsum), SAMSum (samsum), and WLE (wle).
    - [x] Open-ended dialogue system: PersonaChat (pc), DailyDialog (dd), DSTC7-AVSD (da), and SGD (sgd).
    - [x] Data-to-text generation: WebNLG v2.1 (webnlg), WebNLG v3.0 (webnlg2), WikiBio (wikibio), E2E (e2e), DART (dart), and ToTTo (totto).
    - [x] Question generation: SQuAD (squadqg) and CoQA (coqaqg).
    - [x] Story generation: ROCStories (roc) and WritingPrompts (wp).
    - [x] Question answering: SQuAD (squad) and CoQA (coqa).
    - [ ] Task-oriented dialogue system: MultiWOZ 2.0 (multiwoz).
    - [x] Commonsense generation: CommonGen (cg).
    - [x] Text simplification: WikiAuto + Turk/ASSET (wia).
    - [x] Paraphrase generation: Quora (quora) and ParaNMT (paranmt).
    - [x] Text style transfer: GYAFC-E&M (gyafc_em) and F&R (gyafc_fr).


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
    - [ ] Text simplification: WikiAuto + Turk/ASSET (wia). https://arxiv.org/pdf/2005.00352v2.pdf https://arxiv.org/pdf/2110.08329v2.pdf
    - [ ] Paraphrase generation: Quora (comming soon).
    - [ ] Text style transfer: GYAFC-E&M and F&R (comming soon).
