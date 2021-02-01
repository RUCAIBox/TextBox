![TextBox Logo](asset/logo.png)

------

# TextBox (妙笔)

*“李太白少时，梦所用之笔头上生花后天才赡逸，名闻天下。”——王仁裕《开元天宝遗事·梦笔头生花》*

[![PyPi Latest Release](https://img.shields.io/pypi/v/textbox)](https://pypi.org/project/textbox/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Documentation Status](https://readthedocs.org/projects/textbox/badge/?version=latest)](https://textbox.readthedocs.io/en/latest/?badge=latest)
[![Release](https://img.shields.io/github/v/release/rucaibox/textbox.svg)](https://github.com/rucaibox/textbox/releases)

[文档] | [模型] | [数据集] | [论文] | [English Version]

[文档]: https://textbox.readthedocs.io/en/latest/
[模型]: #模型
[数据集]: #数据集
[论文]: https://arxiv.org/abs/2101.02046
[English Version]: README.md

TextBox是基于Python和PyTorch开发的，用于在一个统一的、全面的、高效的框架中复现和开发文本生成算法，主要面向研究者使用。我们的库包括16种文本生成算法，涵盖了两个主要任务：

+ 无条件（无输入）生成
+ 序列到序列（Seq2Seq）生成，包括机器翻译和摘要生成

我们支持6个文本生成的基准数据集，用户可以使用我们的库重新处理原始数据，或者简单地下载我们团队已经处理好的数据集。

<p align="center">
  <img src="asset/framework.png" alt="TextBox v0.1 architecture">
  <br>
  <b>图片</b>: TextBox整体框架
</p>


## 特点

- **统一模块化框架：** TextBox基于PyTorch构建，通过将不同的模型解耦为一组高度可重用的模块，实现高度模块化设计。
- **全面的基准模型、数据集和标准评估：** TextBox还包含广泛的基准文本生成模型，涵盖VAE、GAN、基于RNN或Transformer的模型，以及预训练语言模型（PLM）等类别。
- **高度灵活及拓展性强的框架：** TextBox在文本生成模型部分提供了多种常用的函数与模块的接口。例如RNN encoder-decoder, Transformer encoder-decoder以及各种预训练语言模型。
- **简单便捷的上手：** TextBox提供了灵活的配置文件，生手不用修改源代码即可运行实验，研究者也可以通过修改少量配置文件进行量化实验。

## 安装

TextBox 需要的安装条件:

- `Python >= 3.6.2`

- `torch >= 1.6.0`. 请根据你的CUDA版本和NVIDIA驱动版本参考 [官方指南](https://pytorch.org/get-started/locally/) 来安装合适的版本。

- `GCC >= 5.1.0`

### 通过 pip 安装

```bash
pip install textbox
```

如果你安装`fast_bleu`时遇到了问题，对于Linux请保证`GCC >= 5.1.0`；对于Windows，请使用[fast_bleu_wheel4windows](https://github.com/RUCAIBox/TextBox/tree/main/fast_bleu_wheel4windows)目录下的wheel文件进行安装；对于MacOS，请使用如下命令安装：

```bash
pip install fast-bleu --install-option="--CC=<path-to-gcc>" --install-option="--CXX=<path-to-g++>"
```

成功安装后`fast_bleu`，重新安装`textbox`即可。

### 通过源文件安装

```bash
git clone https://github.com/RUCAIBox/TextBox.git && cd TextBox
pip install -e . --verbose
```

## 快速上手

### 从源文件开始

下载TextBox源码后，你可以使用下述提供的脚本快速上手使用该库：

```bash
python run_textbox.py
```

上述脚本会在COCO数据集上运行RNN模型进行无条件生成。一般而言，这个运行示例会执行几分钟。我们可以获得如 [example.log](asset/example.log) 这样的输出日志。

如果你想要改变运行的参数，如`rnn_type`，`max_vocab_size`，只需要按照你的需要设定额外的命令参数：

```bash
python run_textbox.py --rnn_type=lstm --max_vocab_size=4000
```

我们也支持修改相应数据集和模型 [properties](https://github.com/RUCAIBox/TextBox/tree/main/textbox/properties) 文件夹中的YAML配置文件，并将其包含在命令行中。

如果你想修改模型、数据集或任务类型，只需通过修改相应的命令参数来运行脚本：

```bash
python run_textbox.py --model=[model_name] --dataset=[dataset_name] --task_type=[task_name]
```

`model_name` 是将被运行的模型，比如RNN或者BART。 我们实现了的模型可以在 [模型](#模型) 中找到。

TextBox 包含了三种主要类型的文本生成，分别是`unconditional`（无条件）, `translation`（翻译） and `summarization`（摘要）。

如果你想要修改数据集，请参考 [数据集](#数据集)。

### 从API开始

如果TextBox是由pip安装的，你可以创建一个新的python文件，下载数据集，并编写和运行以下代码：

```python
from textbox.quick_start import run_textbox

run_textbox(config_dict={'model': 'RNN',
                         'dataset': 'COCO',
                         'data_path': './dataset',
                         'task_type': 'unconditional'})
```

这将在COCO数据集上进行RNN模型的训练和测试。

如果你想运行不同的模型、参数或数据集，可以使用与 [从源文件开始](#从源文件开始) 相同的操作。

### **使用预训练语言模型**

TextBox支持部分预训练语言模型进行文本生成任务，下面以GPT-2为例，展示我们如何利用预训练语言模型进行fine-tuning。

1. 从Hugging Face提供的模型源 (https://huggingface.co/gpt2/tree/main) 中下载GPT-2模型，包括`config.json`, `merges.txt`, `pytorch_model.bin`, `tokenizer.json`和`vocab.json`五个文件，将其放在与`textbox`同级的文件夹下，例如`pretrained_model/gpt2`。

2. 下载好模型之后，直接通过脚本运行：

```bash
python run_textbox.py --model=GPT2 --dataset=COCO --task_type=unconditional \
                      --pretrained_model_path=pretrained_model/gpt2
```

## 结构

上述[图片](#textbox-妙笔)展示了TextBox的整体架构。程序的运行需要从文件、命令行或参数字典中获取实验参数配置，数据集和模型会根据设置的配置进行初始化，之后执行模块负责对模型进行训练和评估。获取更多接口相关的细节可以参考[说明文档](https://textbox.readthedocs.io/en/latest/)。

### 模型

我们总共实现了包括无条件生成和sequence-to-sequence生成在内的16个文本生成模型，其中基础的RNN语言模型用于无条件文本生成，另外的15个模型可以参照下表：

<table align="center">
<thead>
<tr>
<th align="center">类别</th>
<th align="center">任务</th>
<th align="center">模型</th>
<th align="center">引用</th>
</tr>
</thead>
<tbody><tr>
<td align="center" rowspan="3"><strong>VAE</strong></td>
<td align="center" rowspan="9"><strong>Unconditional</strong></td>
<td align="center">LSTMVAE</td>
<td align="center"><a href="https://arxiv.org/abs/1511.06349">(Bowman et al., 2016)</a></td>
</tr>
<tr>
<td align="center">CNNVAE</td>
<td align="center"><a href="https://arxiv.org/abs/1702.08139">(Yang et al., 2017)</a></td>
</tr>
<tr>
<td align="center">HybridVAE</td>
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

### 数据集

我们总共收集了6个在上述提及的3类文本生成任务中常用的数据集，这些数据集可以通过[Google Drive](https://drive.google.com/drive/folders/1iNRErGM3YRDF3hjY8DMpWaQo-prmUtNX?usp=sharing) 和 [百度网盘](https://pan.baidu.com/s/1upHl8SXGNjZ2LCfV-L164Q) (密码: lwy6)来下载，数据集中包含原始数据以及处理过的数据。 

在下表我们列出了6个数据集：

<table align="center">
<thead>
<tr>
<th align="center">任务</th>
<th align="center">数据集</th>
</tr>
</thead>
<tbody><tr>
<td align="center" rowspan="3"><strong>Unconditional</strong></td>
<td align="center">Image COCO Caption</td>
</tr>
<tr>
<td align="center">EMNLP2017 WMT News</td>
</tr>
<tr>
<td align="center">IMDB Movie Review</td>
</tr>
<tr>
<td align="center" rowspan="2"><strong>Translation</strong></td>
<td align="center">IWSLT2014 German-English</td>
</tr>
<tr>
<td align="center">WMT2014 English-German</td>
</tr>
<tr>
<td align="center"><strong>Summarization</strong></td>
<td align="center">GigaWord</td>
</tr>
</tbody>
</table>

下载好的数据集需要放到 `dataset` 目录下面，和我们项目中的结构类似。

我们也支持用户在自己的数据集上训练模型，只需要按照下面三个步骤操作即可：

1. 在 `dataset` 目录下面创建一个新的目录用于放置用户自己的数据集，数据集要求每行包含一个文本序列，例如 `dataset/YOUR_DATASET`;

2. 创建一个YAML参数配置文件用于对自己数据集的超参数进行配置，YAML的文件名称应与数据集名称相同，例如 `textbox/properties/dataset/YOUR_DATASET.yaml`. 

   如果你想对数据集进行分割，请在YAML文件中设置 `split_strategy: "load_split"`，具体可以参考 [COCO yaml](/textbox/properties/dataset/COCO.yaml) 或者 [IWSLT14_DE_EN yaml](/textbox/properties/dataset/IWSLT14_DE_EN.yaml).

   如果你想按照比例自动对数据集进行划分，请在YAML文件中设置 `split_strategy: "by_ratio"` 和 `split_ratio` 这两个参数，具体可以参考 [IMDB yaml](/textbox/properties/dataset/IMDB.yaml).

3. 对于无条件文本生成，如果你设置了 `"by_ratio"` ，请将数据集命名为 `corpus.txt` ，如果你设置了  `"load_split"` ，请将数据集命名为 `train.txt, valid.txt, dev.txt` 。

   对于sequence-to-sequence文本生成，我们只支持划分好的数据集。请将数据集命名为 `train.[xx/yy], valid.[xx/yy], dev.[xx/yy]` ， `xx` 或者 `yy` 是源文件或目标文件的后缀，应与YAML文件中的 `source_suffix` 和 `target_suffix` 保持一致。

## 实验结果

我们实现了多个文本生成模型，并在有条件文本生成和无条件文本生成任务上对他们的结果进行了比较。我们也展示了一些生成实例，更多的实例可以[生成实例](https://github.com/RUCAIBox/TextBox/tree/main/generated_examples)找到。

*在前期实验中，我们的TextBox得到了以下结果。然而，这些算法是根据我们的理解和经验来实现和调整的，这可能没有达到它们的最佳性能。如果你能在某个具体算法上得到更好的结果，请告知我们。验证结果后，我们会更新该表。*

### 无条件文本生成

#### Image COCO Caption

测试集负对数似然 (NLL), BLEU and Self-BLEU (SBLEU)度量结果展示：

|     Model     |  NLL  | BLEU-2 | BLEU-3 | BLEU-4 | BLEU-5 | SBLEU-2 | SBLEU-3 | SBLEU-4 | SBLEU-5 |
| :-----------: | :---: | :----: | :----: | :----: | :----: | :-----: | :-----: | :-----: | :-----: |
|  **RNNVAE**   | 33.02 | 80.46  |  51.5  | 25.89  | 11.55  |  89.18  |  61.58  |  32.69  |  14.03  |
|  **CNNVAE**   | 36.61 |  0.63  |  0.27  |  0.28  |  0.29  |  3.10   |  0.28   |  0.29   |  0.30   |
| **HybridVAE** | 56.44 | 31.96  |  3.75  |  1.61  |  1.76  |  77.79  |  26.77  |  5.71   |  2.49   |
|  **SeqGAN**   | 30.56 | 80.15  | 49.88  | 24.95  | 11.10  |  84.45  |  54.26  |  27.42  |  11.87  |
|  **TextGAN**  | 32.46 | 77.47  | 45.74  | 21.57  |  9.18  |  82.93  |  51.34  |  24.41  |  10.01  |
|  **RankGAN**  | 31.07 | 77.36  | 45.05  | 21.46  |  9.41  |  83.13  |  50.62  |  23.79  |  10.08  |
|  **MaliGAN**  | 31.50 | 80.08  | 49.52  | 24.03  | 10.36  |  84.85  |  55.32  |  28.28  |  12.09  |
|  **LeakGAN**  | 25.11 | 93.49  | 82.03  | 62.59  | 42.06  |  89.73  |  64.57  |  35.60  |  14.98  |
|  **MaskGAN**  | 95.93 | 58.07  | 21.22  |  5.07  |  1.88  |  76.10  |  43.41  |  20.06  |  9.37   |
|  **GPT-2**  | 26.82 | 75.51  | 58.87  | 38.22  | 21.66  |  92.78  |  75.47  |  51.74  |  32.39  |

部分生成实例展示：

<table align="center">
<thead>
<tr>
<th align="center">模型</th>
<th align="center">实例</th>
</tr>
</thead>
<tbody><tr>
<td align="center"><strong>RNNVAE</strong></td>
<td>people playing polo to eat in the woods .</td>
</tr>
<tr>
<td align="center"><strong>LeakGAN</strong></td>
<td>a man is standing near a horse on a lush green grassy field .</td>
</tr>
<tr>
<td align="center"><strong>GPT-2</strong></td>
<td>cit a large zebra lays down on the ground.</td>
</tr>
</tbody></table>

#### EMNLP2017 WMT News

测试集NLL, BLEU, SBLEU度量结果展示：

|     Model     |  NLL   | BLEU-2 | BLEU-3 | BLEU-4 | BLEU-5 | SBLEU-2 | SBLEU-3 | SBLEU-4 | SBLEU-5 |
| :-----------: | :----: | :----: | :----: | :----: | :----: | :-----: | :-----: | :-----: | :-----: |
|  **RNNVAE**   | 142.23 | 58.81  | 19.70  |  5.57  |  2.01  |  72.79  |  27.04  |  7.85   |  2.73   |
|  **CNNVAE**   | 164.79 |  0.82  |  0.17  |  0.18  |  0.18  |  2.78   |  0.19   |  0.19   |  0.20   |
| **HybridVAE** | 177.75 | 29.58  |  1.62  |  0.47  |  0.49  |  59.85  |  10.3   |  1.43   |  1.10   |
|  **SeqGAN**   | 142.22 | 63.90  | 20.89  |  5.64  |  1.81  |  70.97  |  25.56  |  7.05   |  2.18   |
|  **TextGAN**  | 140.90 | 60.37  | 18.86  |  4.82  |  1.52  |  68.32  |  23.24  |  6.10   |  1.84   |
|  **RankGAN**  | 142.27 | 61.28  | 19.81  |  5.58  |  1.82  |  67.71  |  23.15  |  6.63   |  2.09   |
|  **MaliGAN**  | 149.93 | 45.00  | 12.69  |  3.16  |  1.17  |  65.10  |  20.55  |  5.41   |  1.91   |
|  **LeakGAN**  | 162.70 | 76.61  | 39.14  | 15.84  |  6.08  |  85.04  |  54.70  |  29.35  |  14.63  |
|  **MaskGAN**  | 303.00 | 63.08  | 21.14  |  5.40  |  1.80  |  83.92  |  47.79  |  19.96  |  7.51   |
|   **GPT-2**   | 88.01  | 55.88  | 21.65  |  5.34  |  1.40  |  75.67  |  36.71  |  12.67  |  3.88   |

部分生成实例展示：

<table align="center">
<thead>
<tr>
<th align="center">模型</th>
<th align="center">实例</th>
</tr>
</thead>
<tbody><tr>
<td align="center"><strong>RNNVAE</strong></td>
<td>lewis holds us in total because they have had a fighting opportunity to hold any bodies when companies on his assault .</td>
</tr>
<tr>
<td align="center"><strong>LeakGAN</strong></td>
<td>we &#39; re a frustration of area , then we do coming out and play stuff so that we can be able to be ready to find a team in a game , but I know how we &#39; re going to say it was a problem .</td>
</tr>
<tr>
<td align="center"><strong>GPT-2</strong></td>
<td>russ i&#39;m trying to build a house that my kids can live in, too, and it&#39;s going to be a beautiful house.</td>
</tr>
</tbody></table>

#### IMDB Movie Review

测试集NLL, BLEU, SBLEU度量结果展示：

|     Model     |  NLL   | BLEU-2 | BLEU-3 | BLEU-4 | BLEU-5 | SBLEU-2 | SBLEU-3 | SBLEU-4 | SBLEU-5 |
| :-----------: | :----: | :----: | :----: | :----: | :----: | :-----: | :-----: | :-----: | :-----: |
|  **RNNVAE**   | 445.55 | 29.14  | 13.73  |  4.81  |  1.85  |  38.77  |  14.39  |  6.61   |  5.16   |
|  **CNNVAE**   | 552.09 |  1.88  |  0.11  |  0.11  |  0.11  |  3.08   |  0.13   |  0.13   |  0.13   |
| **HybridVAE** | 318.46 | 38.65  |  2.53  |  0.34  |  0.31  |  70.05  |  17.27  |  1.57   |  0.59   |
|  **SeqGAN**   | 547.09 | 66.33  | 26.89  |  6.80  |  1.79  |  72.48  |  35.48  |  11.60  |  3.31   |
|  **TextGAN**  | 488.37 | 63.95  | 25.82  |  6.81  |  1.51  |  72.11  |  30.56  |  8.20   |  1.96   |
|  **RankGAN**  | 518.10 | 58.08  | 23.71  |  6.84  |  1.67  |  69.93  |  31.68  |  11.12  |  3.78   |
|  **MaliGAN**  | 552.45 | 44.50  | 15.01  |  3.69  |  1.23  |  57.25  |  22.04  |  7.36   |  3.26   |
|  **LeakGAN**  | 499.57 | 78.93  | 58.96  | 32.58  | 12.65  |  92.91  |  79.21  |  60.10  |  39.79  |
|  **MaskGAN**  | 509.58 | 56.61  | 21.41  |  4.49  |  0.86  |  92.09  |  77.88  |  59.62  |  42.36  |
|   **GPT-2**   | 348.67 | 72.52  | 41.75  | 15.40  |  4.22  |  86.21  |  58.26  |  30.03  |  12.56  |

部分生成实例展示（最大长度 `max_length` 为100）：

<table align="center">
<thead>
<tr>
<th align="center">模型</th>
<th align="center">实例</th>
</tr>
</thead>
<tbody><tr>
<td align="center"><strong>RNNVAE</strong></td>
<td>best brilliant known plot , sound movie , although unfortunately but it also like . the almost five minutes i will have done its bad numbers . so not yet i found the difference from with</td>
</tr>
<tr>
<td align="center"><strong>LeakGAN</strong></td>
<td>i saw this film shortly when I storms of a few concentration one before it all time . It doesn t understand the fact that it is a very good example of a modern day , in the &lt;|unk|&gt; , I saw it . It is so bad it s a &lt;|unk|&gt; . the cast , the stars , who are given a little</td>
</tr>
<tr>
<td align="center"><strong>GPT-2</strong></td>
<td>be a very bad, low budget horror flick that is not worth watching and, in my humble opinion, not worth watching any time. the acting is atrocious, there are scenes that you could laugh at and the story, if you can call it that, was completely lacking in logic and</td>
</tr>
</tbody></table>

### 序列到序列（seq2seq）文本生成

#### IWSLT2014 German-English（翻译）

测试集上的BLEU度量有三种解码策略：top-k采样、贪婪搜索和beam搜索（beam搜索大小 `beam_size` 设置为5）:

<table align="center">
<thead>
<tr>
<th align="center">模型</th>
<th align="center">解码策略</th>
<th align="center">BLEU-2</th>
<th align="center">BLEU-3</th>
<th align="center">BLEU-4</th>
<th align="center">BLEU</th>
</tr>
</thead>
<tbody><tr>
<td align="center" rowspan="3"><b>RNN with Attention</b></td>
<td align="center">Top-k sampling</td>
<td align="center">26.68</td>
<td align="center">16.95</td>
<td align="center">10.85</td>
<td align="center">19.66</td>
</tr>
<tr>
<td align="center">Greedy search</td>
<td align="center">33.74</td>
<td align="center">23.03</td>
<td align="center">15.79</td>
<td align="center">26.23</td>
</tr>
<tr>
<td align="center">Beam search</td>
<td align="center">35.68</td>
<td align="center">24.94</td>
<td align="center">17.42</td>
<td align="center">28.23</td>
</tr>
<tr>
<td align="center" rowspan="3"><b>Transformer</b></td>
<td align="center">Top-k sampling</td>
<td align="center">30.96</td>
<td align="center">20.83</td>
<td align="center">14.16</td>
<td align="center">23.91</td>
</tr>
<tr>
<td align="center">Greedy search</td>
<td align="center">35.48</td>
<td align="center">24.76</td>
<td align="center">17.41</td>
<td align="center">28.10</td>
</tr>
<tr>
<td align="center">Beam search</td>
<td align="center">36.88</td>
<td align="center">26.10</td>
<td align="center">18.54</td>
<td align="center">29.49</td>
</tr>
<tr>
<td align="center"><b>BART</b></td>
<td align="center">Beam search</td>
<td align="center">29.02</td>
<td align="center">19.58</td>
<td align="center">13.48</td>
<td align="center">22.42</td>
</tr>
<td align="center"><b>BERT2BERT</b></td>
<td align="center">Beam search</td>
<td align="center">27.61</td>
<td align="center">18.41</td>
<td align="center">12.46</td>
<td align="center">21.07</td>
</tr>
</tbody></table>


部分生成实例展示：

<table align="center">
<tbody><tr>
<td align="center"><b>源语言（德语）</b></td>
<td>wissen sie , eines der großen &lt; unk &gt; beim reisen und eine der freuden bei der &lt; unk &gt; forschung ist , gemeinsam mit den menschen zu leben , die sich noch an die alten tage erinnern können . die ihre vergangenheit noch immer im wind spüren , sie auf vom regen &lt; unk &gt; steinen berühren , sie in den bitteren blättern der pflanzen schmecken .</td>
</tr>
<tr>
<td align="center"><b>真实目标语言（英语）</b></td>
<td>you know , one of the intense pleasures of travel and one of the delights of &lt; unk &gt; research is the opportunity to live amongst those who have not forgotten the old ways , who still feel their past in the wind , touch it in stones &lt; unk &gt; by rain , taste it in the bitter leaves of plants .</td>
</tr>
<tr>
<td align="center"><b>RNN with Attention</b></td>
<td>you know , one of the great &lt; unk &gt; trips is a travel and one of the friends in the world &amp; apos ; s investigation is located on the old days that you can remember the past day , you &amp; apos ; re &lt; unk &gt; to the rain in the &lt; unk &gt; chamber of plants .</td>
</tr>
<tr>
<td align="center"><b>Transformer</b></td>
<td>you know , one of the great &lt; unk &gt; about travel , and one of the pleasure in the &lt; unk &gt; research is to live with people who remember the old days , and they still remember the wind in the wind , but they &amp; apos ; re touching the &lt; unk &gt; .</td>
</tr>
</tbody></table>

#### GigaWord （摘要）

使用beam搜索在测试集上的ROUGE度量（beam搜索大小 `beam_size` 设置为5）:

<table align="center">
<thead>
<tr>
<th align="center">Model</th>
<th align="center">ROUGE-1</th>
<th align="center">ROUGE-2</th>
<th align="center">ROUGE-L</th>
<th align="center">ROUGE-W</th>
</tr>
</thead>
<tbody><tr>
<td align="center"><strong>RNN with Attention</strong></td>
<td align="center">36.32</td>
<td align="center">17.63</td>
<td align="center">38.36</td>
<td align="center">25.08</td>
</tr>
<tr>
<td align="center"><strong>Transformer</strong></td>
<td align="center">36.21</td>
<td align="center">17.64</td>
<td align="center">38.10</td>
<td align="center">24.89</td>
</tr>
<tr>
<td align="center"><strong>BART</strong></td>
<td align="center">39.34</td>
<td align="center">20.07</td>
<td align="center">41.25</td>
<td align="center">27.13</td>
</tr>
<tr>
<td align="center"><strong>BERT2BERT</strong></td>
<td align="center">38.16</td>
<td align="center">18.89</td>
<td align="center">40.06</td>
<td align="center">26.21</td>
</tr>
</tbody></table>


部分生成实例展示：

<table align="center">
<tbody><tr>
<td align="center"><b>文章</b></td>
<td>japan 's nec corp. and computer corp. of the united states said wednesday they had agreed to join forces in supercomputer sales .
</td>
</tr>
<tr>
<td align="center"><b>真实摘要</b></td>
<td>nec in computer sales tie-up</td>
</tr>
<tr>
<td align="center"><b>RNN with Attention</b></td>
<td>nec computer corp .</td>
</tr>
<tr>
<td align="center"><b>Transformer</b></td>
<td>nec computer to join forces in chip sales</td>
</tr>
</tbody></table>

## TextBox重要发布

| 发行版本 |    日期    |   特点    |
| :------: | :--------: | :-----------: |
|  v0.1.5  | 2021/01/11 | Basic TextBox |

## 贡献

如果您遇到错误或有任何建议，请通过 [filing an issue](https://github.com/RUCAIBox/TextBox/issues).

我们欢迎关于修复错误、添加新特性的任何贡献。

我们希望所有的贡献者先在issue中提出问题，然后再提PR。

## 引用

如果你觉得TextBox对你的科研工作有帮助，请引用我们的 [论文](https://arxiv.org/abs/2101.02046):

```
@article{recbole,
    title={TextBox: A Unified, Modularized, and Extensible Framework for Text Generation},
    author={Junyi Li, Tianyi Tang, Gaole He, Jinhao Jiang, Xiaoxuan Hu, Puzhao Xie, Wayne Xin Zhao, Ji-Rong Wen},
    year={2021},
    journal={arXiv preprint arXiv:2101.02046}
}
```

## 项目团队

TextBox 由 [AI Box](http://aibox.ruc.edu.cn/) 团队成员开发。

## 许可
TextBox 使用 [MIT License](./LICENSE)。
