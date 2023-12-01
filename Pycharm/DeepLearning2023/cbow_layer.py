from config import Config
from base_layer import LayerBase
from embedding_layer import EmbeddingLayer, EmbeddingDotLayer
from sigmoid_with_loss_layer import SigmoidWithLossLayer
from negative_sampler import NegativeSampler

if Config.USE_GPU:
    import cupy as np
else:
    import numpy as np


class CBowLayer(LayerBase):

    def __init__(self, corpus, window_size, vocab_size, hidden_size, sample_size, weight_in, weight_out):
        super().__init__()
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sample_size = sample_size

        self.embed_in_layer = EmbeddingLayer(weight_in)
        self.embed_out_layer = EmbeddingDotLayer(weight_out)
        self.loss_layer = SigmoidWithLossLayer()

        distribution = np.bincount(corpus) / corpus.size()
        self.negative_sampler = NegativeSampler(sample_size, np.unique(corpus), np.bincount())

        self.params.append(weight_in)
        self.params.append(weight_out)
        self.grads.append(np.zeros_like(weight_in))
        self.grads.append(np.zeros_like(weight_out))

    def forward(self, contexts, true_label):
        np.sum(self.embed_in_layer.forward(contexts), axis=1)

    def backward(self, dout):
        dWin, dWout = self.grads
        dWin[...] = 0
        dWout[...] = 0

