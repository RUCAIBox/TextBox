![TextBox Logo](asset/logo.png)

------

# TextBox (文骏)

[![PyPi Latest Release](https://img.shields.io/pypi/v/textbox)](https://pypi.org/project/textbox/) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

[Docs] | [Datasets] | [Paper]

[Docs]: https://rucaibox.github.io/textbox.github.io/
[Datasets]: https://github.com/RUCAIBox/RecDatasets
[Paper]: https://arxiv.org/abs/2011.01731

TextBox is developed based on Python and PyTorch for reproducing and developing text generation algorithms in a unified, comprehensive and efficient framework for research purpose. Our library includes 16 text generation algorithms, covering two major tasks:

+ Unconditional (input-free) Generation
+ Sequence to Sequence (Seq2Seq) Generation, including Machine Translation and Summarization

We provide the support for 6 benchmark text generation datasets. A user can apply our library to process the original data copy, or simply download the processed datasets by our team.
<p align="center">
  <img src="asset/framework.png" alt="TextBox v0.1 architecture">
  <br>
  <b>Figure</b>: TextBox Overall Architecture
</p>

## Feature

- **Unified and modularized framework.** TextBox is built upon PyTorch and designed to be highly modularized, by decoupling diverse models into a set of highly reusable modules.
- **Comprehensive models, benchmark datasets and standardized evaluations.** TextBox also contains a wide range of text generation models, covering the categories of VAE, GAN, RNN or Transformer based models, and pre-trained language models (PLM).
- **Extensible and flexible framework.** TextBox provides convenient interfaces of various common functions or modules in text generation models, RNN encoder-decoder, Transformer encoder-decoder and pre-trained language model.

## Installation

TextBox requires:

- `Python >= 3.6.2`

- `torch >= 1.6.0`. Please follow the [official instructions](https://pytorch.org/get-started/locally/) to install the appropriate version according to your CUDA version and NVIDIA driver version.

- `GCC >= 5.1.0`

### Install from pip

```bash
pip install textbox
```

### Install from source
```bash
git clone https://github.com/RUCAIBox/TextBox.git && cd TextBox
pip install -e . --verbose
```

## Quick-Start

### Start from source

With the source code, you can use the provided script for initial usage of our library:

```bash
python run_textbox.py --model=RNN --dataset=COCO --task_type=unconditional
```

This script will run the RNN model on the COCO dataset. Typically, this example takes a few minutes. We will obtain the output log like [example.log](asset/example.log).

If you want to change the parameters, such as `rnn_type`, `max_vocab_size`, just set the additional command parameters as you need:

```bash
python run_textbox.py --model=RNN --dataset=COCO --task_type=unconditional \
                      --rnn_type=lstm --max_vocab_size=4000
```

We also support to modify YAML configuration files in corresponding dataset and model [properties](/tree/main/textbox/properties) folders and include it in the command line.

If you want to change the model, the dataset or the task type, just run the script by modifying corresponding command parameters: 

```bash
python run_textbox.py --model=[model_name] --dataset=[dataset_name] --task_type=[task_name]
```

`model_name` is the model to be run, such as RNN and BART. Models we implemented and their details can be found in [Model](/#Model).

TextBox covers three major task types of text generation, namely `unconditional`, `translation` and `summarization`.

If you want to change the datasets, please refer to [Data](/#Data).

### Start from API

If TextBox is installed from pip, you can create a new python file, download the dataset, and write and run the following code:

```python
from textbox.quick_start import run_textbox

run_textbox(config_dict={'model': 'RNN',
                         'dataset': 'COCO',
                         'data_path': './dataset',
                         'task_type': 'unconditional'})
```

This will perform the training and test of the RNN model on the COCO dataset.

If you want to run different models, parameters or datasets, the operations are same with [Start from source](#Start-from-source).

## Architecture

### Model



### Dataset





## Experiment Results



## Reference

If you find TextBox useful for your research or development, please cite the following [paper](https://arxiv.org/abs/2011.01731):

```
@article{recbole,
    title={TextBox: A Unified, Modularized, and Extensible Framework for Text Generation},
    author={Junyi Li, Tianyi Tang, Gaole He, Jinhao Jiang, Xiaoxuan Hu, Puzhao Xie, Wayne Xin Zhao, Ji-Rong Wen},
    year={2021},
    journal={arXiv preprint arXiv:2011.01731}
}
```

## The Team

TextBox is developed and maintained by [AI Box](http://aibox.ruc.edu.cn/).

## License
TextBox uses [MIT License](./LICENSE).

