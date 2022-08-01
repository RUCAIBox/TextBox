![TextBox Logo](asset/logo.png)

---

# TextBox (妙笔)

*“李太白少时，梦所用之笔头上生花后天才赡逸，名闻天下。”——王仁裕《开元天宝遗事·梦笔头生花》*

[![PyPi Latest Release](https://img.shields.io/pypi/v/textbox)](https://pypi.org/project/textbox/)
[![Release](https://img.shields.io/github/v/release/rucaibox/textbox.svg)](https://github.com/rucaibox/textbox/releases)
[![Documentation Status](https://readthedocs.org/projects/textbox/badge/?version=latest)](https://textbox.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

[Docs] | [Model] | [Dataset]

[Docs]: https://textbox.readthedocs.io/en/latest/
[Model]: #Model
[Dataset]: #Datase

TextBox is developed based on Python and PyTorch for reproducing and developing text generation algorithms in a unified, comprehensive and efficient framework for research purpose. Our library includes 21 text generation algorithms, covering two major tasks:

- Unconditional (input-free) Generation
- Conditional (Seq2Seq) Generation, including Machine Translation, Text Summarization, Attribute-to-Text, and Dialogue Systems

We provide the support for 9 benchmark text generation datasets. A user can apply our library to process the original data copy, or simply download the processed datasets by our team.

<!-- ===================== Installation ===================== -->

## Installation

```python
pip install textbox
```

> **Note**
>
> **TroubleShooting**: If you face a problem when installing `fast-bleu`, for Linux, please ensure `GCC >= 5.1.0`. For Windows, you can use the wheels in [fast_bleu_wheel4windows](https://github.com/RUCAIBox/TextBox/tree/main/fast_bleu_wheel4windows) for installation. For MacOS, you can install with the following command: `pip install fast-bleu --install-option="--CC=$(which gcc-11)" --install-option="--CXX=$(which g++-11)"`.

### Install from Source

```bash
git clone https://github.com/RUCAIBox/TextBox.git && cd TextBox
bash install.sh
```

### W&B Dashboard Configuration

Weights&Biases dashboard is intergrated. For the first run, follow the prompt to register an account and log in with [API key](https://wandb.ai/authorize). See [advanced configuration](#wb-dashboard-advanced-configuration) for more information.

<!-- ===================== Quick Start ===================== -->

## Quick Start

The script below will run the `BART` model on the `samsum` dataset. The yielded files mainly include a log file like [example.log](asset/example.log) in `log` and checkpoint files in `saved`. See [Model Parameters](#model-parameters) for more detail of model_path.

```bash
python run_textbox.py --model_path=facebook/bart-base
```

### Specify Model, Dataset, and Evaluation Metrics

Substitute `<xxx>` with your choices. See [Model](#Model), [Dataset](#Dataset), and [Metrics](#Metrics) for a full support list.

```bash
python run_textbox.py ... --model=<model-name> --dataset=<dataset-name> --metrics=<list-of-metrics>
# Example (equivalent of default configuration):
python run_textbox.py --model_path=facebook/bart-base --model=BART --dataset=samsum --metrics=['bleu']
```

> **Warning**
> Backslashes and no-extra-space are required when inputting a list of string like `\[\'bleu\',\'rouge\'\]` in command line. As a result, a preset run configuration as is follows is more recommended.

### Load From YAML and Python Scripts

You may also want to load configuration with `--config_files` from YAML files like [overall.yaml](textbox/properties/overall.yaml), which will be loaded automatically as default values of most of parameters below:

```bash
python run_textbox.py ... --config_files <yaml-file-one> <yaml-file-two>
```

Or start from python scripts:

```python
from textbox.quick_start import run_textbox
run_textbox(config_dict={ 'model': 'BART', 'dataset': 'samsum', ... })
```

### Partial Experiment

Running partial experiment with `do_train`, `do_valid`, `do_test`, and `quick_test=<amount-of-data-to-load>` may help with specialized application, like debugging. In some cases, `load_experiment=<path-to-checkpoint>` is needed to load model beforehand.

```bash
python run_textbox.py ... --do_train=False --load_experiment=example.pth --quick_test=16
```

<!-- ===================== Training ===================== -->

## Training

### Basics

`optimizer=<optimizer-name>` and `scheduler=<scheduler-name>` provides a wrapper around **pytorch optimizer**, which means parameters like `epsilon` or `warmup_steps` can be specified with keyword dictionaries `optimizer_kwargs={ 'epsilon': ... }` and `scheduler_kwargs={ 'warmup_steps': ... }`. See [pytorch optimizer](https://pytorch.org/docs/stable/optim.html#algorithms) and [scheduler]() for a complete tutorial.  <!-- TODO -->

Variable validation pace is introduced to validate the model **at each specific batch-steps or epochs**. Specify `valid_strategy` (either `'step'` or `'epoch'`) and `valid_intervals=<int>` to adjust the pace. Specifically, traditional train-validate paradigm is a special case with `valid_strategy=epoch` and `valid_intervals=1`.

`max_save=<int>` indicates **the maximal amount of saved files** (checkpoint and generated corpus during evaluation). `-1`: save every file, `0`: do not save any file, `1`: only save the file with best score, and `n`: save both the best and the last $n−1$ files.

**Early stopping** can be configured with `metrics_for_best_model=<list-of-metrics-entries>`, which is used to calculate score, and `stopping_steps=<int>`, which specifies the amount of validation steps:

```bash
python run_textbox.py ... --stopping_steps=8 --metrics_for_best_model=\[\'rouge-1\', \'rouge-w\'\]
```

or yaml equivalent:

```yaml
stopping_steps: 8
metrics_for_best_model: ['rouge-1', 'rouge-w']
```

Other commonly used parameters includes `epochs=<int>` and `max_steps=<int>` (indicating maximal iteration of epochs and batch steps), `learning_rate=<float>`, `train_batch_size=<int>`, `weight_decay=<bool>`, and `grad_clip=<bool>`.

#### Pre-trained Model Parameters

`model_path` receives a name of model on [huggingface](https://huggingface.co/models) like [`facebook/bart-base`](https://huggingface.co/models?search=facebook%2Fbart-base).

Not only `model_path`, but `config_path` and `tokenizer_path` (same value with `model_path` by default) also receive a huggingface model or a local path. Besides, `config_kwargs` and `tokenizer_kwargs` are useful when additional parameters are required.

For example, when building a *Task-oriented Dialogue System*, special tokens can be added with `additional_special_tokens`; fast tokenization can also be switched with `use_fast`:

```yaml
config_kwargs: {}
tokenizer_kwargs: { 'use_fast': False, 'additional_special_tokens': ['[db_0]', '[db_1]', '[db_2]'] }
```

Other commonly used parameters includes `label_smoothing`

```yaml
label_smoothing: <smooth-loss-weight>
```

The full keyword arguments should be found in [PreTrainedTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) or documents of corresponding tokenizer.

##### Generation Parameters

Pre-trained model is able to perform generation using various methods by combining different [parameters](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate). By default, beam search is adapted:

```yaml
generation_kwargs: {'num_beams': 5, 'early_stopping': True}
```

> **Note**
> This `early_stopping` is different from [the one](#Basics) in training.

Nucleus sampling is also supported by pre-trained model:

```yaml
generation_kwargs: {'do_sample': True, 'top_k': 10, 'top_p': 0.9}
```

#### Dataset Parameters

`src_len`, `tgt_len`, and `truncate` restricts the maximal length of source/target sentence and the positional to be truncated (`head` or `tail`). For some models used for translation task like m2m100, you need to specify source language and target language:

```yaml
# m2m100: en -> zh
src_lang: 'en'
tgt_lang: 'zh'
```

#### Evaluation Parameters

After specifying several evaluation metrics, further configuration on them is as follows:

For example, `rouge` provides `rouge_max_ngrams` and `rouge_type` to specify the maximal number of ngrams and type of rouge (like `files2rouge`, `rouge-score`, etc.). In addition, `bleu` provides `bleu_max_ngrams`, `bleu_type`, `smoothing_function=<int>`, and `corpus_bleu=<bool>` to customize metric.

```yaml
bleu_max_ngrams: 4
bleu_type: nltk
smoothing_function: 0
corpus_bleu: False

distinct_max_ngrams: 4
```

Other evaluation metrics ([full list](#Evaluation)) observe the same naming rules.

### Prompting

To prompt at `prefix` or `suffix`, pass strings to the following parameters:

```yaml
prefix_prompt: 'Summarize: '
suffix_prompt: ' (Write a story)'
```

#### Parameter-efficient Prompting

Besides human instruction, parameter-efficient prompting is also supported，though only for `BART`, `T5`, and `GPT-2` model, with methods including `lora`, `prefix-tuning` (equivalent to `p-tuning-v2`), `adapter`, and `prompt-tuning`:

```yaml
efficient_methods: ['adapter', 'prompt-tuning']
```

To further modify the prompting methods, use `efficient_unfreeze_model` and `efficient_kwargs` by finding parameters for corresponding methods in [docs]() and put them together in keyword arguments:  <!-- TODO -->

```yaml
efficient_kwargs: { 'adapter_mid_dim': <int>, 'prompt_length': <int> }
efficient_unfreeze_model: <bool>
```

### Pretraining

Pretraining models from scratch or continue pretraining from existing checkpoints is essential to achieving SOTA results. We support modularized pretraining tasks as individual collate functions to meet flexible pretraining demands. 

Currently, we support pretraining tasks from BART paper, including Text-Infilling and Sentence Permutation (which is sufficient to reproduce BART results from the original paper according to this [Github Issue](https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320) from BART author).

To enable pretraining, simply set `--pretrain_task` to `denoising` or `text_infilling`(by default, pretraining is disabled and thus set to `disabled`). We plan to add more pretraining tasks at `textbox/data/utils.py`.

```bash
python run_textbox.py ... --pretrain_task=<task-name>
```


### Efficient Training

#### Distributed Data Parallel

TextBox supports to train models with multiple GPUs conveniently. You don't need to modify the model, just run the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=<gpu-num> \
       run_textbox.py ... --gpu_id=<gpu-ids> --DDP=True
```

`gpu_num` is the number of GPUs you want to train with (such as 4), and `gpu_ids` is the usable GPU id list (such as `\[0,1,2,3\]`).

> **Note**
> Only end-to-end model is supported for DDP by now. Non-end-to-end models such as GAN are coming soon.

#### FP16 Accelerate

Configurate and test (not always necessary) [accelerate](https://github.com/huggingface/accelerate) with `accelerate config` and `accelerate test` in shell ([example](asset/accelerate.md)).

To accelerate multi-gpu training with `accelerate`, run the script below. Note that `accelerate` occupies a communication port, to resolve port conflict, an available port have to be allocated with `main_process_port` manually.

```bash
accelerate launch [--main_process_port <port-number>] run_textbox.py ...
```

### Hyper-Parameters Tuning

```bash
python run_hyper.py --space=textbox/properties/hyperopt_example.test --algo='exhaustive'  --model_path=facebook/bart-base --metrics=\[\'rouge\''\] --metrics_for_best_model=\[\'ROUGE-1\'\]
```

A separate script `run_hyper.py` is provided for hyper-parameters tuning. Use `space=<path-to-space-file>` and `algo=<algo-name>` to select from different configurations ([tutorial](textbox/asset/hyper_tuning.md).

### Multiple Random Seeds

Similar to hyper-parameters tuning, another python code with new parameter `multi_seed=<int>` indicating amount of seeds to be test, is introduced for multiple random seeds test:

```bash
python run_multi_seed.py --multi_seed=16  --model_path=facebook/bart-base --metrics=\[\'rouge\''\] --metrics_for_best_model=\[\'ROUGE-1\'\]
```

Specify `seed` parameter to reproduce generation of multiple seeds.

### W&B Dashboard Advanced Configuration

If you are running your code in jupyter environments, you may want to login by simply setting an environment variable (your key may be stored in plain text):

```python
%env WANDB_API_KEY=<your-key>
```

If you are debugging your model, you may want to **disable W&B** with `wandb disabled` in the command line and **none of the metrics** will be recorded. To re-enable it, use `wandb enabled`.

You can also disable **sync only** with `wandb offline` and enable it again with `wandb online`. The local files can be uploaded by executing `wandb sync`.


<!-- ===================== Model ===================== -->


## Model

<!-- Thanks for table generatros https://www.tablesgenerator.com/html_tables -->


<div class="tg-wrap"><table align="center">
<thead>
  <tr>
    <th align="center">Category</th>
    <th align="center">Model</th>
    <th align="center">Label Name</th>
    <th align="center">Reference</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="6" align="center"><strong>CLM</strong></td>
    <td align="center">CPM</td>
    <td align="center">CPM</td>
    <td align="center"><a href="https://arxiv.org/pdf/2012.00413">(Zhang et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center">CTRL</td>
    <td align="center">CTRL</td>
    <td align="center"><a href="https://arxiv.org/pdf/1909.05858">(Keskar et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center">GPT</td>
    <td align="center">OpenAI-GPT</td>
    <td align="center"><a href="https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf">(Radford et al., 2018)</a></td>
  </tr>
  <tr>
    <td align="center">GPT Neo</td>
    <td align="center">GPT_neo</td>
    <td align="center"><a href="https://arxiv.org/pdf/2101.00027">(Gao et al., 2021)</a></td>
  </tr>
  <tr>
    <td align="center">GPT2</td>
    <td align="center">GPT2</td>
    <td align="center"><a href="https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf">(Radford et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center">OPT</td>
    <td align="center">OPT</td>
    <td align="center"><a href="https://arxiv.org/pdf/2205.01068.pdf">(Artetxe et al., 2022)</a></td>
  </tr>
  <tr>
    <td rowspan="16" align="center"><strong>Seq2Seq</strong></td>
    <td align="center">BART</td>
    <td align="center">BART</td>
    <td align="center"><a href="https://arxiv.org/pdf/1910.13461">(Lewis et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center">Bert2Bert</td>
    <td align="center">Bert2Bert</td>
    <td align="center"><a href="https://aclanthology.org/2020.tacl-1.18.pdf">(Rothe et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center">BigBirdPegasus</td>
    <td align="center">BigBird-Pegasus</td>
    <td align="center"><a href="https://arxiv.org/pdf/2007.14062">(Zaheer et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center">Blenderbot</td>
    <td align="center">Blenderbot</td>
    <td rowspan="2"><a href="https://arxiv.org/pdf/2004.13637.pdf">(Roller et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center">Blenderbot Small</td>
    <td align="center">Blenderbot-Small</td>
  </tr>
  <tr>
    <td align="center">LED</td>
    <td align="center">LED</td>
    <td align="center"><a href="https://arxiv.org/pdf/2004.05150">(Beltagy et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center">M2M100</td>
    <td align="center">M2M_100</td>
    <td align="center"><a href="https://arxiv.org/pdf/2010.11125">(Fan et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center">MBart</td>
    <td align="center">MBart</td>
    <td align="center"><a href="https://arxiv.org/pdf/2001.08210">(Liu et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center">mT5</td>
    <td align="center">mT5</td>
    <td align="center"><a href="https://arxiv.org/pdf/2010.11934">(Xue et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center">MVP</td>
    <td align="center">MVP</td>
    <td align="center"><a href="https://arxiv.org/pdf/2206.12131">(Tang et al., 2022)</a></td>
  </tr>
  <tr>
    <td align="center">Pegasus</td>
    <td align="center">Pegasus</td>
    <td align="center"><a href="https://arxiv.org/pdf/1912.08777">(Zhang et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center">ProphetNet</td>
    <td align="center">ProphetNet</td>
    <td align="center"><a href="https://arxiv.org/pdf/2001.04063">(Qi et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center">T5</td>
    <td align="center">T5</td>
    <td align="center"><a href="https://arxiv.org/pdf/1910.10683.pdf">(Raffel et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center">Chinese BART</td>
    <td align="center">Chinese-BART</td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">Chinese Pegasus</td>
    <td align="center">Chinese-Pegasus</td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">CPT</td>
    <td align="center">CPT</td>
    <td align="center"><a href="https://arxiv.org/pdf/2109.05729">(Shao et al., 2021)</a></td>
  </tr>
</tbody>
</table></div>


<!-- ===================== Dataset ===================== -->


## Dataset

Now we support 11 generation tasks and corresponding datasets:
- Text summarization: CNN/Daily Mail (cnndm), XSum (xsum), SAMSum (samsum), and WLE (wle).
- Open-ended dialogue system: PersonaChat (pc), DailyDialog (dd), DSTC7-AVSD (da), and SGD (sgd).
- Data-to-text generation: WebNLG v2.1 (webnlg), WebNLG v3.0 (webnlg2), WikiBio (wikibio), E2E (e2e), DART (dart), and ToTTo (totto).
- Question generation: SQuAD (squadqg) and CoQA (coqaqg).
- Story generation: ROCStories (roc) and WritingPrompts (wp).
- Question answering: SQuAD (squad) and CoQA (coqa).
- Task-oriented dialogue system: MultiWOZ 2.0 (multiwoz).
- Commonsense generation: CommonGen (cg).
- Text simplification: WikiAuto + Turk/ASSET (wia).
- Paraphrase generation: Quora (comming soon).
- Text style transfer: GYAFC-E&M and F&R (comming soon).

We also support you to run our model using your own dataset. Just follow the three steps:

1. Create a new folder under the `dataset` folder to put your own corpus file which includes a sequence per line, e.g. `dataset/YOUR_DATASET`;
2. Write a YAML configuration file using the same file name to set the hyper-parameters of your dataset, e.g. `textbox/properties/dataset/YOUR_DATASET.yaml`.

   If you want to splitted the dataset, please set `split_strategy: "load_split"` in the yaml, just as the [COCO yaml](/textbox/properties/dataset/COCO.yaml) or [IWSLT14_DE_EN yaml](/textbox/properties/dataset/IWSLT14_DE_EN.yaml).

   If you want to split the dataset by ratio automaticly, please set `split_strategy: "by_ratio"` and your desired `split_ratio` in the yaml, just as the [IMDB yaml](/textbox/properties/dataset/IMDB.yaml).
3. For unconditional generation, name the corpus file `corpus.txt` if you set`"by_ratio"`, name the corpus files `train.txt, valid.txt, dev.txt` if you set `"load_split"`.

   For sequence-to-sequence generation, please name the corpus files `train.[xx/yy], valid.[xx/yy], dev.[xx/yy]`, and the `xx` or `yy` is the suffix of the source or target file which should be consistent with `source_suffix` and `target_suffix` in the YAML.


<!-- ===================== Evaluation ===================== -->


## Evaluation

15 mainstream evaluation metrics are intergrated:

<div class="tg-wrap"><table>
<thead>
  <tr>
    <th colspan="5" align="center">Evaluation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">bert_score</td>
    <td align="center">bleu</td>
    <td align="center">chrf</td>
    <td align="center">chrf+</td>
    <td align="center">chrf++</td>
  </tr>
  <tr>
    <td align="center">cider</td>
    <td align="center">distinct</td>
    <td align="center">meteor</td>
    <td align="center">nist</td>
    <td align="center">qa</td>
  </tr>
  <tr>
    <td align="center">rouge</td>
    <td align="center">self_bleu</td>
    <td align="center">spice</td>
    <td align="center">ter</td>
    <td align="center">unique</td>
  </tr>
</tbody>
</table></div>

<!-- ===================== Experiment Results ===================== -->


## Experiment Results

<!-- TODO -->


<!-- ===================== Other ===================== -->


## Releases

<!-- TODO -->

| Releases |    Date    |    Features   |
| :------: | :--------: | :-----------: |
|  v1.0.0  | 30/06/2022 |      Test     |
|  v0.2.1  | 15/04/2021 |    TextBox    |
|  v0.1.5  | 01/11/2021 | Basic TextBox |

## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/RUCAIBox/TextBox/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.

We thank [@LucasTsui0725](https://github.com/LucasTsui0725/) for contributing HRED model and [@Richar-Du](https://github.com/Richar-Du/) for CVAE model.

We thank [@wxDai](https://github.com/Dai-Wenxun) for contributing PointerNet and more than 20 language models in transformers API.

We thank [@sbrodeur](https://github.com/sbrodeur) for [code](https://github.com/hyperopt/hyperopt/issues/200#issuecomment-507287308) of exhaustive search for hyper tuning.  <!-- TODO -->

## Reference

If you find TextBox useful for your research or development, please cite the following [paper](https://arxiv.org/abs/2101.02046):

```
@article{textbox,
    title={TextBox: A Unified, Modularized, and Extensible Framework for Text Generation},
    author={Junyi Li, Tianyi Tang, Gaole He, Jinhao Jiang, Xiaoxuan Hu, Puzhao Xie, Wayne Xin Zhao, Ji-Rong Wen},
    year={2021},
    journal={arXiv preprint arXiv:2101.02046}
}
```

## The Team

TextBox is developed and maintained by [AI Box](http://aibox.ruc.edu.cn/).

## License

TextBox uses [MIT License](./LICENSE).

