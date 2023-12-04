from base_layer import LayerBase
from word2vec.embedding_layer import EmbeddingLayer, EmbeddingDotLayer
from word2vec.sigmoid_with_loss_layer import SigmoidWithLossLayer
from word2vec.negative_sampler import NegativeSampler
from utils import np


class CBowLayer(LayerBase):

    def __init__(self, corpus, vocab_size, hidden_size, sample_size, weight_in, weight_out):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sample_size = sample_size

        self.embed_in_layer = EmbeddingLayer(weight_in)
        self.embed_dot_layer = EmbeddingDotLayer(weight_out)
        self.loss_layer = SigmoidWithLossLayer()

        distribution = np.bincount(corpus, minlength=vocab_size) / len(corpus)
        self.negative_sampler = NegativeSampler(np.unique(corpus), distribution)

        self.params.append(weight_in)
        self.params.append(weight_out)
        self.grads.append(self.embed_in_layer.grads[0])
        self.grads.append(self.embed_dot_layer.grads[0])

        self.cache = None

    def forward(self, contexts, true_target):
        batch_size = true_target.shape[0]

        embed_contexts = self.embed_in_layer.forward(contexts)
        h = np.sum(embed_contexts, axis=1) / contexts.shape[1]
        self.cache = contexts.shape[1]

        samples, sample_labels = self.negative_sampler.get_mixed_samples_and_labels(self.sample_size, true_target)
        sample_scores = self.embed_dot_layer.forward(samples, h)

        sample_loss = self.loss_layer.forward(sample_scores, sample_labels)
        return np.sum(sample_loss, axis=1)

    def backward(self, dout):
        dsample_loss = np.repeat(dout[np.newaxis].T, self.sample_size, axis=1)
        dscore = self.loss_layer.backward(dsample_loss)
        dh = self.embed_dot_layer.backward(dscore)

        context_size = self.cache
        dembed_sum = dh/context_size
        dembed_context = np.repeat(dembed_sum[:, np.newaxis, :], context_size, axis=1)
        self.embed_in_layer.backward(dembed_context)
