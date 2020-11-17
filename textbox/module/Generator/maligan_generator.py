# @Time   : 2020/11/11
# @Author : Xiaoxuan Hu
# @Email  : huxiaoxuan@ruc.edu.cn


from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder


class MaliGANGenerator(BasicRNNDecoder):
    def __init__(self, embedding_size, hidden_size, num_layers, rnn_type, dropout_ratio):
        super(MaliGANGenerator, self).__init__(embedding_size, hidden_size, num_layers, rnn_type, dropout_ratio)