![TextBox Logo](asset/logo.png)

------

# TextBox (文骏)

[![PyPi Latest Release](https://img.shields.io/pypi/v/textbox)](https://pypi.org/project/textbox/)[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

[Docs] | [Datasets] | [Paper]

[Docs]: https://rucaibox.github.io/textbox.github.io/
[Datasets]: https://github.com/RUCAIBox/RecDatasets
[Paper]: https://arxiv.org/abs/2011.01731

TextBox is developed based on Python and PyTorch for reproducing and developing text generation algorithms in a unified, comprehensive and efficient framework for research purpose. Our library includes 16 text generation algorithms, covering two major tasks:

+ Unconditional (input-free) Generation
+ Sequence to Sequence (Seq2Seq) Generation

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

## The Team

TextBox is developed and maintained by [AI Box](http://aibox.ruc.edu.cn/).

## License
TextBox uses [MIT License](./LICENSE).
