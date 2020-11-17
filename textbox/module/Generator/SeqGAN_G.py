import torch
import torch.nn as nn
import torch.nn.functional as F

from textbox.model.LM.rnn import RNN


class SeqGANGenerator(RNN):
    def __init__(self, config, dataset):
        super(SeqGANGenerator, self).__init__(config, dataset)