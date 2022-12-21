![TextBox Logo](asset/logo.png)

---

# TextBox 2.0 (妙笔)

*“李太白少时，梦所用之笔头上生花后天才赡逸，名闻天下。”——王仁裕《开元天宝遗事·梦笔头生花》*

TextBox 2.0 is an up-to-date text generation library based on Python and PyTorch focusing on building a unified and standardized pipeline for applying Pre-trained language models to text generation:

- From a **task** perspective, we consider 13 common text generation tasks such as translation, story generation and style transfer, and their corresponding 83 widely-used datasets. 
- From a **model** perspective, we incorporate 47 PLMs/modules covering the categories of general, translation, dialogue, controllable, distilled, Chinese, and light-weight PLMs.
- From a **training** perspective, we support 4 pre-training objectives and 4 efficient and robust training strategies, such as distributed data parallel and efficient generation.


Compared with the previous version of TextBox, this extension mainly focuses on building a unified, flexible and standardized framework for better supporting PLM-based text generation models. There are three advantages in TextBox 2.0:

- It is a significant innovation focusing on comprehensive tasks and PLMs.
- It is designed to be unified in implementation and interface.
- It produces very similar performance with original or official implementations.

<p align="center">
  <img src="asset/framework.png" alt="TextBox 2.0 framework" width="50%" height="50%">
  <br>
  The Overall Framework of TextBox 2.0
</p>

<!-- ===================== Installation ===================== -->

## Installation

```bash
git clone https://github.com/RUCAIBox/TextBox.git && cd TextBox
bash install.sh
```

## Quick Start

The script below will run the facebook `BART-base` model on the `samsum` dataset. 

```bash
python run_textbox.py --model_path=facebook/bart-base
```

Substitute `--model=<xxx>` ,  `--dataset=<xxx>` and `--model_path=<xxx>` with your choices. See [Model](https://github.com/RUCAIBox/TextBox/blob/2.0.0/asset/model.md#model-list), [Dataset](https://github.com/RUCAIBox/TextBox/blob/2.0.0/asset/dataset.md#dataset-list) for a full support list. See [Pre-trained Model Parameters](https://github.com/RUCAIBox/TextBox/blob/2.0.0/asset/model.md#pre-trained-model-parameters) for more detail of `model_path`.


```bash
python run_textbox.py --model=<model-name> --dataset=<dataset-name> --model_path=<hf-or-local-path> ...
# Example (equivalent of default configuration):
python run_textbox.py --model=BART --dataset=samsum --model_path=facebook/bart-base
```

<!-- ===================== Training ===================== -->

## Training

### Basic Training

For basic training, we provide a detailed tutorial([here](asset/basic_training.md)) for setting commonly used parameters like optimizer, scheduler, validation frequency, early stopping, and so on.

### Pre-training

TextBox 2.0 provides four pre-training objectives to help users pre-train a model from scratch, including language modeling, masked sequence-to-sequence modeling, denoising auto-encoding, and masked span prediction. See [pre-training doc](asset/pretaining.md) for a detailed tutorial.

### Efficient Training

Four useful training methods are provided for improving the optimization of PLMs: distributed data parallel, efficient decoding, hyper-parameter optimization, and repeated experiments. Detailed instructions are provided [here](asset/efficient_training.md).


<!-- ===================== Model ===================== -->

## Model

To support the rapid progress of PLMs on text generation, TextBox 2.0 incorporates 47 Models/Modules, covering the categories of general, translation, Chinese, dialogue, controllable, distilled, prompting, and lightweight models (modules). See [model doc](asset/model.md) for information on detailed [usage instructions of each model/module](https://github.com/RUCAIBox/TextBox/blob/2.0.0/asset/model.md#model-list), [pre-trained model parameters](https://github.com/RUCAIBox/TextBox/blob/2.0.0/asset/model.md#pre-trained-model-parameters), and [generation parameters](https://github.com/RUCAIBox/TextBox/blob/2.0.0/asset/model.md#generation-parameters).

<!-- ===================== Dataset ===================== -->


## Dataset

Now we support 13 generation tasks (e.g., translation and story generation) and their corresponding 83 datasets. We also provide the description, basic statistics, training/validation/testing samples and leaderboard for each dataset. See more details [here](asset/dataset.md).

We also support you to run our model using your own dataset. 

<details>
<summary>Usage</summary>


1. Create a new folder under the `dataset` folder to put your own corpus file which includes a sequence per line, e.g. `dataset/YOUR_DATASET`;
2. Write a YAML configuration file using the same file name to set the hyper-parameters of your dataset, e.g. `textbox/properties/dataset/YOUR_DATASET.yaml`.

</details>

<!-- ===================== Evaluation ===================== -->

## Evaluation

TextBox 2.0 supports 17 automatic metrics of 4 categories and several visualization tools to explore and analyze the generated texts in various dimensions. For evaluation details, see the [evaluation doc](asset/evaluation.md).


## Releases

<!-- TODO -->

| Releases |    Date    |   Features    |
| :------: | :--------: | :-----------: |
|  v2.0.0  | 20/08/2022 |    TextBox    |
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

