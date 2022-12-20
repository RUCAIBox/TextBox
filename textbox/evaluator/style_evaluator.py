from .abstract_evaluator import AbstractEvaluator
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from transformers import BartTokenizerFast


class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embeding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        y = self.embeding(x)
        return y


class TextCNN(nn.Module):

    def __init__(self):
        super(TextCNN, self).__init__()

        embed_dim = 300
        vocab_size = 50265
        num_filters = [128, 128, 128, 128, 128]
        filter_sizes = [1, 2, 3, 4, 5]
        dropout = 0.5

        self.feature_dim = sum(num_filters)
        self.embeder = EmbeddingLayer(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            self.dropout, nn.Linear(self.feature_dim, int(self.feature_dim / 2)), nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), 2)
        )

    def forward(self, inp):
        inp = self.embeder(inp).unsqueeze(1)
        convs = [F.relu(conv(inp)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        out = torch.cat(pools, 1)
        logit = self.fc(out)
        return logit


class StyleEvaluator(AbstractEvaluator):
    r"""Style Evaluator. Now, we support metrics `'style'`
    """

    def __init__(self, config):
        super(StyleEvaluator, self).__init__(config)
        self.max_len = config['tgt_len']
        self.batch_size = config['eval_batch_size']
        self.device = config['device']
        self.tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large', add_prefix_space=True)

        self.model = TextCNN()
        self.model.to(self.device).eval()
        self.model.load_state_dict(torch.load(f'textbox/evaluator/utils/{config["dataset"]}.ckpt'))

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.ngrams
        """
        results = {}
        num = 0
        with torch.no_grad():
            for i in range(ceil(len(generate_corpus.text) / self.batch_size)):
                texts = generate_corpus.text[i * self.batch_size:(i + 1) * self.batch_size]
                input_ids = self.tokenizer(
                    texts, max_length=self.max_len, padding=True, truncation=True, return_tensors="pt"
                )['input_ids'].to(self.device)
                logits = self.model(input_ids)
                num += logits.argmax(dim=-1).sum().item()
        results['style'] = num / len(generate_corpus.text) * 100
        return results
