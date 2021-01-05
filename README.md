![TextBox Logo](asset/logo.png)

------

# TextBox (文骏)

[![PyPi Latest Release](https://img.shields.io/pypi/v/textbox)](https://pypi.org/project/textbox/) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

[Docs] | [Model] | [Dataset] | [Paper]

[Docs]: https://rucaibox.github.io/textbox.github.io/
[Model]: #Model
[Dataset]: #Dataset
[Paper]: https://arxiv.org/abs/2011.01731

TextBox is developed based on Python and PyTorch for reproducing and developing text generation algorithms in a unified, comprehensive and efficient framework for research purpose. Our library includes 16 text generation algorithms, covering two major tasks:

+ Unconditional (input-free) Generation
+ Sequence-to-Sequence (Seq2Seq) Generation, including Machine Translation and Summarization

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

`model_name` is the model to be run, such as RNN and BART. Models we implemented can be found in [Model](#Model).

TextBox covers three major task types of text generation, namely `unconditional`, `translation` and `summarization`.

If you want to change the datasets, please refer to [Dataset](#Dataset).

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

The above [Figure](#TextBox (文骏)) presents the overall architecture of our library. The running procedure relies on some experimental configuration, obtained from the files, command line or parameter dictionaries. The dataset and model are prepared and initialized according to the configured settings, and the execution module is responsible for training and evaluating models. The details of interfaces can be obtained in our [document](https://rucaibox.github.io/textbox.github.io/).

### Model

We implement 16 text generation models covering unconditional generation and sequence-to-sequence generation. We include the basic RNN language model for unconditional generation, and the remaining 15 models in the following table:

<table>
<thead>
<tr>
<th align="center">Category</th>
<th align="center">Task Type</th>
<th align="center">Model</th>
<th align="center">Reference</th>
</tr>
</thead>
<tbody><tr>
<td align="center" rowspan="3"><strong>VAE</strong></td>
<td align="center" rowspan="9"><strong>Unconditional</strong></td>
<td align="center">LSTM-VAE</td>
<td align="center"><a href="https://arxiv.org/abs/1511.06349">(Bowman et al., 2016)</a></td>
</tr>
<tr>
<td align="center">CNN-VAE</td>
<td align="center"><a href="https://arxiv.org/abs/1702.08139">(Yang et al., 2017)</a></td>
</tr>
<tr>
<td align="center">Hybrid-VAE</td>
<td align="center"><a href="https://arxiv.org/abs/1702.02390">(Semeniuta et al., 2017)</a></td>
</tr>
<tr>
<td align="center" rowspan="6"><strong>GAN</strong></td>
<td align="center">SeqGAN</td>
<td align="center"><a href="https://arxiv.org/abs/1609.05473">(Yu et al., 2017)</a></td>
</tr>
<tr>
<td align="center">TextGAN</td>
<td align="center"><a href="https://arxiv.org/abs/1706.03850">(Zhang et al., 2017)</a></td>
</tr>
<tr>
<td align="center">RankGAN</td>
<td align="center"><a href="https://arxiv.org/abs/1705.11001">(Lin et al., 2017)</a></td>
</tr>
<tr>
<td align="center">MaliGAN</td>
<td align="center"><a href="https://arxiv.org/abs/1702.07983">(Che et al., 2017)</a></td>
</tr>
<tr>
<td align="center">LeakGAN</td>
<td align="center"><a href="https://arxiv.org/abs/1709.08624">(Guo et al., 2018)</a></td>
</tr>
<tr>
<td align="center">MaskGAN</td>
<td align="center"><a href="https://arxiv.org/abs/1801.07736">(Fedus et al., 2018)</a></td>
</tr>
<tr>
<td align="center" rowspan="6"><strong>Seq2Seq</strong></td>
<td align="center" rowspan="6"><strong>Translation<br></b><br></b>Summarization</strong></td>
<td align="center">RNN</td>
<td align="center"><a href="https://arxiv.org/abs/1409.3215">(Sutskever et al., 2014)</a></td>
</tr>
<tr>
<td align="center">Transformer</td>
<td align="center"><a href="https://arxiv.org/abs/1706.03762">(Vaswani et al., 2017b)</a></td>
</tr>
<tr>
<td align="center">GPT-2</td>
<td align="center"><a href="https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf">(Radford et al.)</a></td>
</tr>
<tr>
<td align="center">XLNet</td>
<td align="center"><a href="https://arxiv.org/abs/1906.08237">(Yang et al., 2019)</a></td>
</tr>
<tr>
<td align="center">BERT2BERT</td>
<td align="center"><a href="https://arxiv.org/abs/1907.12461">(Rothe et al., 2020)</a></td>
</tr>
<tr>
<td align="center">BART</td>
<td align="center"><a href="https://arxiv.org/abs/1910.13461">(Lewis et al., 2020)</a></td>
</tr>
</tbody></table>
The provided hyper-parameters, APIs and details of our model can be found in our [document](https://rucaibox.github.io/textbox.github.io/).

### Dataset

We have also collected 6 datasets that are commonly used for above three tasks, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj?usp=sharing) and [Baidu Wangpan](https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g) (Password: e272), including raw data and processed data. 

We list the 6 datasets along with their download source or script in the following table:

<style>.lay_fix{table-layout:fixed}</style>
<style>.wid20{width:20%}</style>
<style>.wid60{width:60%}</style>
<table class="lay_fix">
<thead>
<tr>
<th class="wid20">Task</th>
<th class="wid20">Dataset</th>
<th class="wid60">Downloaded Source</th>
</tr>
</thead>
<tbody><tr>
<td align="center" rowspan="3"><strong>Unconditional</strong></td>
<td align="center">Image COCO Caption</td>
<td style="word-break:break-all;"><a href="https://github.com/pclucas14/GansFallingShort/tree/master/real_data_experiments/data/coco">https://github.com/pclucas14/GansFallingShort/tree/master/real_data_experiments/data/coco</a></td>
</tr>
<tr>
<td align="center">EMNLP2017 WMT News</td>
<td style="word-break:break-all;"><a href="https://github.com/pclucas14/GansFallingShort/tree/master/real_data_experiments/data/news">https://github.com/pclucas14/GansFallingShort/tree/master/real_data_experiments/data/news</a></td>
</tr>
<tr>
<td align="center">IMDB Movie Review</td>
<td style="word-break:break-all;"><a href="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz">https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz</a></td>
</tr>
<tr>
<td align="center" rowspan="2"><strong>Translation</strong></td>
<td align="center">IWSLT2014 German-English</td>
<td style="word-break:break-all;"><a href="https://github.com/facebookarchive/MIXER/blob/master/prepareData.sh">https://github.com/facebookarchive/MIXER/blob/master/prepareData.sh</a></td>
</tr>
<tr>
<td align="center">WMT2014 English-German</td>
<td style="word-break:break-all;"><a href="https://github.com/terranceliu/fairseq/blob/efficient_decoding/examples/translation/prepare-wmt14en2de.sh">https://github.com/terranceliu/fairseq/blob/efficient_decoding/examples/translation/prepare-wmt14en2de.sh</a></td>
</tr>
<tr>
<td align="center"><strong>Summarization</strong></td>
<td align="center">GigaWord</td>
<td style="word-break:break-all;"><a href="https://github.com/microsoft/unilm/tree/master/unilm-v1#abstractive-summarization---gigaword">https://github.com/microsoft/unilm/tree/master/unilm-v1#abstractive-summarization---gigaword</a></td>
</tr>
</tbody></table>

The downloaded dataset should be placed in the `dataset` folder, just as our main branch.

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

