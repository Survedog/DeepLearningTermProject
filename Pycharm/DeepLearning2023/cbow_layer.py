from base_layer import LayerBase
from embedding_layer import EmbeddingLayer, EmbeddingDotLayer
from sigmoid_with_loss_layer import SigmoidWithLossLayer
from negative_sampler import NegativeSampler
from utils import np


class CBowLayer(LayerBase):

    def __init__(self, corpus, window_size, vocab_size, hidden_size, sample_size, weight_in, weight_out):
        super().__init__()
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sample_size = sample_size

        self.embed_in_layer = EmbeddingLayer(weight_in)
        self.embed_dot_layer = EmbeddingDotLayer(weight_out)
        self.loss_layer = SigmoidWithLossLayer()

        distribution = np.bincount(corpus, minlength=vocab_size) / len(corpus)
        self.negative_sampler = NegativeSampler(sample_size, np.unique(corpus), distribution)

        self.params.append(weight_in)
        self.params.append(weight_out)
        self.grads.append(np.zeros_like(weight_in))
        self.grads.append(np.zeros_like(weight_out))

    def forward(self, contexts, positive_label):
        batch_size = positive_label.shape[0]

        embed_contexts = self.embed_in_layer.forward(contexts)
        h = np.sum(embed_contexts, axis=1) / (self.window_size * 2)

        negative_samples = self.negative_sampler.get_negative_samples(self.sample_size - 1, positive_label)
        targets, target_bool_labels = np.hstack((positive_label.reshape(batch_size, 1), negative_samples))
        score = self.embed_dot_layer.forward(targets, h)

        self.loss_layer.forward(score, target_bool_labels)

    def backward(self, dout):
        dWin, dWout = self.grads[0], self.grads[1]
        dWin[...] = 0
        dWout[...] = 0
        return dWin, dWout
