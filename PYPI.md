# TextBox (妙笔)

TextBox is developed based on Python and PyTorch for reproducing and developing text generation algorithms in a unified, comprehensive and efficient framework for research purpose. Our library includes 21 text generation algorithms, covering two major tasks:

+ Unconditional (input-free) Generation
+ Conditional (Seq2Seq) Generation, including Machine Translation, Text Summarization, Attribute-to-Text, and Dialogue Systems

We provide the support for 9 benchmark text generation datasets. A user can apply our library to process the original data copy, or simply download the processed datasets by our team. 

**View TextBox homepage for more information: [https://github.com/RUCAIBox/TextBox](https://github.com/RUCAIBox/TextBox).**

**View TextBox document for more implement details: [https://textbox.readthedocs.io/en/latest/](https://textbox.readthedocs.io/en/latest/).**


## Feature

- **Unified and modularized framework.** TextBox is built upon PyTorch and designed to be highly modularized, by decoupling diverse models into a set of highly reusable modules.
- **Comprehensive models, benchmark datasets and standardized evaluations.** TextBox also contains a wide range of text generation models, covering the categories of VAE, GAN, RNN or Transformer based models, and pre-trained language models (PLM).
- **Extensible and flexible framework.** TextBox provides convenient interfaces of various common functions or modules in text generation models, RNN encoder-decoder, Transformer encoder-decoder and pre-trained language model.
- **Easy and convenient to get started.** TextBox provides flexible configuration files, which allows green hands to run experiments without modifying source code, and allows researchers to conduct qualitative analysis by modifying few configurations.

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
python run_textbox.py
```

This script will run the RNN model on the COCO dataset to conduct unconditional generation.

If you want to change the parameters, such as `rnn_type`, `max_vocab_size`, just set the additional command parameters as you need:

```bash
python run_textbox.py --rnn_type=lstm --max_vocab_size=4000
```

We also support to modify YAML configuration files in corresponding dataset and model `properties` folders and include it in the command line.

If you want to change the model, the dataset or the task type, just run the script by modifying corresponding command parameters: 

```bash
python run_textbox.py --model=[model_name] --dataset=[dataset_name]
```

`model_name` is the model to be run, such as RNN and BART.

### Start from API

If TextBox is installed from pip, you can create a new python file, download the dataset, and write and run the following code:

```python
from textbox.quick_start import run_textbox

run_textbox(config_dict={'model': 'RNN',
                         'dataset': 'COCO',
                         'data_path': './dataset'})
```

This will perform the training and test of the RNN model on the COCO dataset.

If you want to run different models, parameters or datasets, the operations are same with **Start from source**.

### **Use Pretrained Language Model**

TextBox supports to apply part of pretrained language models (PLM) to conduct text generation. Take the GPT-2 for example, we will show you how to use PLMs to fine-tune.

1. Download the GPT-2 model provided from Hugging Face (https://huggingface.co/gpt2/tree/main), including `config.json`, `merges.txt`, `pytorch_model.bin`, `tokenizer.json`and `vocab.json`. Then put them in a folder at the same level as `textbox`, such as `pretrained_model/gpt2`.

2. After downloading, you just need to run the command:

```bash
python run_textbox.py --model=GPT2 --dataset=COCO \
                      --pretrained_model_path=pretrained_model/gpt2
```

### **Train with Distributed Data Parallel**

TextBox supports to train models with multiple GPUs conveniently. You don't need to modify the model, just run the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=[gpu_num] \
       run_textbox.py --model=[model_name] \
       --dataset=[dataset_name] --gpu_id=[gpu_ids] --DDP=True
```

`gpu_num` is the number of GPUs you want to train with (such as 4), and `gpu_ids` is the usable GPU id list (such as 0,1,2,3).

Notice that: we only support DDP for end-to-end model. We will add support for non-end-to-end models, such as GAN, in the future.

## The Team

TextBox is developed and maintained by [AI Box](http://aibox.ruc.edu.cn/).

## License
TextBox uses [MIT License](./LICENSE).

