# Model

### Model List

To support the rapid progress of PLMs on text generation, TextBox 2.0 incorporates 47 models/modules. The following table lists the name and reference of each model/module. Click the model/module name for detailed [usage instructions](https://github.com/RUCAIBox/TextBox/tree/2.0.0/instructions).

<!-- Thanks for table generatros https://www.tablesgenerator.com/html_tables -->

<div class="tg-wrap"><table align="center">
<thead>
  <tr>
    <th align="center" colspan="2">Category</th>
    <th align="center">Model Name</th>
    <th align="center">Reference</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="16" align="center"><strong>General</strong></td>
    <td rowspan="4" align="center"><strong>CLM</strong></td>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/CLM/OpenAI-GPT.md">OpenAI-GPT</td>
    <td align="center"><a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf">(Radford et al., 2018)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/CLM/GPT2.md">GPT2</td>
    <td align="center"><a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">(Radford et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/CLM/GPT_Neo.md">GPT_Neo</td>
    <td align="center"><a href="https://arxiv.org/pdf/2101.00027">(Gao et al., 2021)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/CLM/OPT.md">OPT</td>
    <td align="center"><a href="https://arxiv.org/pdf/2205.01068">(Artetxe et al., 2022)</a></td>
  </tr>
  <tr>
    <td rowspan="12" align="center"><strong>Seq2Seq</strong></td>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/BART.md">BART</td>
    <td align="center"><a href="https://arxiv.org/pdf/1910.13461">(Lewis et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/T5.md">T5</td>
    <td align="center"><a href="https://arxiv.org/pdf/1910.10683">(Raffel et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/UniLM.md">UniLM</td>
    <td align="center"><a href="https://arxiv.org/pdf/1905.03197">(Dong et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/MASS.md">MASS</td>
    <td align="center"><a href="https://arxiv.org/pdf/1905.02450">(Song et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/Pegasus.md">Pegasus</td>
    <td align="center"><a href="https://arxiv.org/pdf/1912.08777">(Zhang et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/ProphetNet.md">ProphetNet</td>
    <td align="center"><a href="https://arxiv.org/pdf/2001.04063">(Qi et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/MVP.md">MVP</td>
    <td align="center"><a href="https://arxiv.org/pdf/2206.12131">(Tang et al., 2022)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/BERT2BERT.md">BERT2BERT</td>
    <td align="center"><a href="https://aclanthology.org/2020.tacl-1.18.pdf">(Rothe et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/BigBird-Pegasus.md">BigBird-Pegasus</td>
    <td align="center"><a href="https://arxiv.org/pdf/2007.14062">(Zaheer et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/LED.md">LED</td>
    <td align="center"><a href="https://arxiv.org/pdf/2004.05150">(Beltagy et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/longT5.md">LongT5</td>
    <td align="center"><a href="https://arxiv.org/pdf/2112.07916">(Guo et al., 2021)</a></td>
  </tr>
  <tr>
      <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Seq2Seq/Pegasus_X.md">PegasusX</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2208.04347">(Phang et al., 2022)</a></td>
  </tr>



  <tr>
    <td rowspan="8" colspan="2" align="center"><strong>Multilingual Models</strong></td>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Multilingual%20Models/mBART.md">mBART</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2001.08210">(Liu et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Multilingual%20Models/mT5.md">mT5</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2010.11934">(Xue et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Multilingual%20Models/Marian.md">Marian</a></td>
    <td align="center"><a href="https://aclanthology.org/2020.eamt-1.61.pdf">(Tiedemann et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Multilingual%20Models/M2M_100.md">M2M_100</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2010.11125">(Fan et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Multilingual%20Models/NLLB.md">NLLB</a></td>
    <td align="center"><a href="https://arxiv.org/ftp/arxiv/papers/2207/2207.04672.pdf">(NLLB Team, 2022)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Multilingual%20Models/XLM.md">XLM</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/1901.07291">(Lample et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Multilingual%20Models/XLM-RoBERTa.md">XLM-RoBERTa</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/1911.02116">(Conneau et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Multilingual%20Models/XLM-ProphetNet.md">XLM-ProphetNet</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2001.04063">(Qi et al., 2020)</a></td>
  </tr>

  <tr>
  <td rowspan="6" colspan="2" align="center"><strong>Chinese Models</strong></td>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Chinese%20Models/CPM.md">CPM</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2012.00413">(Zhang et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Chinese%20Models/CPT.md">CPT</a></td>
    <td align="center" rowspan="2"><a href="https://arxiv.org/pdf/2109.05729">(Shao et al., 2021)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Chinese%20Models/Chinese-BART.md">Chinese-BART</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Chinese%20Models/Chinese-GPT2.md">Chinese-GPT2</a></td>
    <td align="center" rowspan="3"><a href="https://arxiv.org/pdf/1909.05658">(Zhao et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Chinese%20Models/Chinese-T5.md">Chinese-T5</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Chinese%20Models/Chinese-Pegasus.md">Chinese-Pegasus</a></td>
  </tr>

  <tr>
    <td rowspan="3" colspan="2" align="center"><strong>Dialogue Models</strong></td>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Dialogue%20Models/Blenderbot.md">Blenderbot</a></td>
    <td align="center" rowspan="2"><a href="https://arxiv.org/pdf/2004.13637">(Roller et al., 2020)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Dialogue%20Models/Blenderbot-Small.md">Blenderbot-Small</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Dialogue%20Models/DialoGPT.md">DialoGPT</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/1911.00536">(Zhang et al., 2019)</a></td>
  </tr>

  <tr>
    <td rowspan="2" colspan="2" align="center"><strong>Conditional Models</strong></td>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Conditional%20Models/CTRL.md">CTRL</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/1909.05858">(Keskar et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Conditional%20Models/PPLM.md">PPLM</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/1912.02164">(Dathathri et al., 2019)</a></td>
  </tr>

  <tr>
    <td rowspan="2" colspan="2" align="center"><strong>Distilled Models</strong></td>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Distilled%20Models/DistilGPT2.md">DistilGPT2</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/1910.01108">(Sanh et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Distilled%20Models/DistilBART.md">DistilBART</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2010.13002">(Shleifer et al., 2020)</a></td>
  </tr>

  <tr>
    <td rowspan="2" colspan="2" align="center"><strong>Prompting Models</strong></td>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Prompting%20Models/PTG.md">PTG</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2205.01543">(Li et al., 2022a)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Prompting%20Models/Context-Tuning.md">Context-Tuning</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2201.08670">(Tang et al., 2022)</a></td>
  </tr>

  <tr>
  <td rowspan="6" colspan="2" align="center"><strong>Lightweight Modules</strong></td>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Lightweight%20Modules/Adapter.md">Adapter</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/1902.00751">(Houlsby et al., 2019)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Lightweight%20Modules/Prefix-tuning.md">Prefix-tuning</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2101.00190">(Li and Liang, 2021)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Lightweight%20Modules/Prompt%20tuning.md">Prompt tuning</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2104.08691">(Lester et al., 2021)</a></td>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Lightweight%20Modules/LoRA.md">LoRA</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2106.09685">(Hu et al., 2021)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Lightweight%20Modules/BitFit.md">BitFit</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2106.10199">(Ben-Zaken et al. ,2021)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/Lightweight%20Modules/P-Tuning%20v2.md">P-Tuning v2</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/2110.07602">(Liu et al., 2021a)</a></td>
  </tr>

  <tr>
    <td rowspan="2" colspan="2" align="center"><strong>Non-Pre-training Models</strong></td>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/RNN.md">RNN</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/1409.3215">(Sutskever et al., 2014)</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/RUCAIBox/TextBox/blob/2.0.0/instructions/transformer.md">Transformer</a></td>
    <td align="center"><a href="https://arxiv.org/pdf/1706.03762">(Vaswani et al., 2017b)</a></td>
  </tr>

</tbody>
</table></div>

### Pre-trained Model Parameters

TextBox 2.0 is compatible with Hugging Face, so `model_path` receives a name of model on [Hugging Face](https://huggingface.co/models) like [`facebook/bart-base`](https://huggingface.co/models?search=facebook%2Fbart-base) or just a local path.  `config_path` and `tokenizer_path` (same value as `model_path` by default) also receive a Hugging Face model or a local path. 

Besides, `config_kwargs` and `tokenizer_kwargs` are useful when additional parameters are required.

For example, when building a *Task-oriented Dialogue System*, special tokens can be added with `additional_special_tokens`; fast tokenization can also be switched with `use_fast`:

```bash
config_kwargs: {}
tokenizer_kwargs: {'use_fast': False, 'additional_special_tokens': ['[db_0]', '[db_1]', '[db_2]'] }
```

Other commonly used parameters include `label_smoothing: <smooth-loss-weight>`

The full keyword arguments should be found in [PreTrainedTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) or documents of the corresponding tokenizer.

### Generation Parameters

The pre-trained model can perform generation using various methods by combining different [parameters](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate). By default, beam search is adapted:

```bash
generation_kwargs: {'num_beams': 5, 'early_stopping': True}
```

Nucleus sampling is also supported by pre-trained model:

```bash
generation_kwargs: {'do_sample': True, 'top_k': 10, 'top_p': 0.9}
```

