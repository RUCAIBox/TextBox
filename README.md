![TextBox Logo](asset/logo.png)

---

# TextBox 2.0 (妙笔)

*“李太白少时，梦所用之笔头上生花后天才赡逸，名闻天下。”——王仁裕《开元天宝遗事·梦笔头生花》*

[*TextBox 2.0: A Text Generation Library with Pre-trained Language Models*](https://arxiv.org/abs/2212.13005)

TextBox 2.0 is an up-to-date text generation library based on Python and PyTorch focusing on building a unified and standardized pipeline for applying pre-trained language models to text generation:

- From a **task** perspective, we consider 13 common text generation tasks such as translation, story generation, and style transfer, and their corresponding 83 widely-used datasets. 
- From a **model** perspective, we incorporate 47 pre-trained language models/modules covering the categories of general, translation, Chinese, dialogue, controllable, distilled, prompting, and lightweight models (modules).
- From a **training** perspective, we support 4 pre-training objectives and 4 efficient and robust training strategies, such as distributed data parallel and efficient generation.


Compared with the previous version of TextBox, this extension mainly focuses on building a unified, flexible, and standardized framework for better supporting PLM-based text generation models. There are three advantages of TextBox 2.0:

- It is a significant innovation focusing on comprehensive tasks and PLMs.
- It is designed to be unified in implementation and interface.
- It can faithfully reproduce the results reported in existing work.


<p align="center">
  <img src="asset/framework.png" alt="TextBox 2.0 framework" width="50%" height="50%">
  <br>
  The Overall Framework of TextBox 2.0
</p>


<!-- ===================== Installation ===================== -->

## Installation

Considering that a modified version of transformers will be installed, it is recommended to create a new conda environment:
```bash
conda create -n TextBox python=3.8
```

Then, you can clone our repository and install it with one-click.
```bash
git clone https://github.com/RUCAIBox/TextBox.git && cd TextBox
bash install.sh
```

If you face a issue `ROUGE-1.5.5.pl - XML::Parser dependency error` when installing `files2rouge`, you can refer to this [issue](https://github.com/pltrdy/files2rouge/issues/9).

## Quick Start

This is a script template to run TextBox 2.0 in an end-to-end pipeline:

```bash
python run_textbox.py --model=<model-name> --dataset=<dataset-name> --model_path=<hf-or-local-path>
```

Substitute `--model=<xxx>` ,  `--dataset=<xxx>` and `--model_path=<xxx>` with your choices.

The choices of `model` and `model_path` can be found in [Model](asset/model.md#model-list). We provide the detailed instruction of each model in that page.

The choices of `dataset` can be found in [Dataset](asset/dataset.md#dataset-list). You should download the dataset at [https://huggingface.co/RUCAIBox](https://huggingface.co/RUCAIBox) and put the downloaded dataset under the `dataset` folder just like [samsum](https://github.com/RUCAIBox/TextBox/tree/2.0.0/dataset/samsum). If your want to use your own dataset, please refer to [here](asset/dataset.md#new-dataset).


The script below will run the Facebook `BART-base` model on the `samsum` dataset:
```bash
python run_textbox.py --model=BART --dataset=samsum --model_path=facebook/bart-base
```

<!-- ===================== Training ===================== -->

## Training

### Basic Training

For basic training, we provide a detailed tutorial ([here](asset/basic_training.md)) for setting commonly used parameters like optimizer, scheduler, validation frequency, early stopping, and so on. 

### Pre-training

TextBox 2.0 provides four pre-training objectives to help users pre-train a model from scratch, including language modeling, masked sequence-to-sequence modeling, denoising auto-encoding, and masked span prediction. See the [pre-training doc](asset/pretraining.md) for a detailed tutorial.

### Efficient Training

Four useful training methods are provided for improving the optimization of PLMs: distributed data parallel, efficient decoding, hyper-parameter optimization, and repeated experiments. Detailed instructions are provided [here](asset/efficient_training.md).


<!-- ===================== Model ===================== -->

## Model

To support the rapid progress of PLMs on text generation, TextBox 2.0 incorporates 47 models/modules, covering the categories of general, translation, Chinese, dialogue, controllable, distilled, prompting, and lightweight models (modules). See the [model doc](asset/model.md) for information on detailed [usage instructions of each model](asset/model.md#model-list), [pre-trained model parameters](asset/model.md#pre-trained-model-parameters), and [generation parameters](asset/model.md#generation-parameters).

<!-- ===================== Dataset ===================== -->


## Dataset

Now we support 13 generation tasks (e.g., translation and story generation) and their corresponding 83 datasets. We also provide the description, basic statistics, training/validation/testing samples, and leaderboard for each dataset. See more details [here](asset/dataset.md).


<!-- ===================== Evaluation ===================== -->

## Evaluation

TextBox 2.0 supports 17 automatic metrics of 4 categories and several visualization tools to explore and analyze the generated texts in various dimensions. For evaluation details, see the [evaluation doc](asset/evaluation.md).


## Releases


| Releases |    Date    |   Features           |
| :------: | :--------: | :------------------: |
|  v2.0.1  | 24/12/2022 |    TextBox 2.0       |
|  v2.0.0  | 20/08/2022 |    TextBox 2.0 Beta  |
|  v0.2.1  | 15/04/2021 |    TextBox           |
|  v0.1.5  | 01/11/2021 | Basic TextBox        |

## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/RUCAIBox/TextBox/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.

We thank [@LucasTsui0725](https://github.com/LucasTsui0725/) for contributing HRED model and several evaluation metrics.

We thank [@wxDai](https://github.com/Dai-Wenxun) for contributing PointerNet and more than 20 language models in transformers API.

## The Team

TextBox is developed and maintained by [AI Box](http://aibox.ruc.edu.cn/).

## License

TextBox uses [MIT License](./LICENSE).

## Reference

If you find TextBox 2.0 useful for your research or development, please cite the following papers:

```
@inproceedings{tang-etal-2022-textbox,
    title = "{T}ext{B}ox 2.0: A Text Generation Library with Pre-trained Language Models",
    author = "Tang, Tianyi  and  Li, Junyi  and  Chen, Zhipeng  and  Hu, Yiwen  and  Yu, Zhuohao  and  Dai, Wenxun  and  Zhao, Wayne Xin  and  Nie, Jian-yun  and  Wen, Ji-rong",
    booktitle = "Proceedings of the The 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-demos.42",
    pages = "435--444",
}


@inproceedings{textbox,
    title = "{T}ext{B}ox: A Unified, Modularized, and Extensible Framework for Text Generation",
    author = "Li, Junyi  and  Tang, Tianyi  and  He, Gaole  and  Jiang, Jinhao  and  Hu, Xiaoxuan  and  Xie, Puzhao  and  Chen, Zhipeng  and  Yu, Zhuohao  and  Zhao, Wayne Xin  and  Wen, Ji-Rong",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-demo.4",
    doi = "10.18653/v1/2021.acl-demo.4",
    pages = "30--39",
}
```
