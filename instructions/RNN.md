## RNN

You can train a RNN encoder-decoder with attention from scratch with this model. Three models are available:
* RNN
* GRU
* LSTM

You can choose them through ``model=RNN``,``model=GRU``,``model=LSTM``. Meanwhile, you can check or modify the default parameters of the model in ``textbox/property/model/rnn.yaml(gru.yaml)(lstm.yaml)``

Example usage:

```bash
python run_textbox.py \
    --model=RNN \
    --dataset=samsum
```