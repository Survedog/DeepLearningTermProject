from common.base_layer import LayerBase
from word2vec.embedding_layer import EmbeddingLayer, EmbeddingDotLayer
from common.loss_layers import SigmoidWithLossLayer
from word2vec.negative_sampler import NegativeSampler
from common.utils import py


class SkipgramModel(LayerBase):

    def __init__(self, corpus, vocab_size, hidden_size, sample_size, weight_in, weight_out):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sample_size = sample_size
        self.context_size = None

        self.embed_in_layer = EmbeddingLayer(weight_in)
        self.embed_dot_layer = EmbeddingDotLayer(weight_out)
        self.loss_layer = SigmoidWithLossLayer()

        distribution = py.bincount(corpus, minlength=vocab_size) / len(corpus)
        self.negative_sampler = NegativeSampler(py.unique(corpus), distribution)

        self.params.append(weight_in)
        self.params.append(weight_out)
        self.grads.append(self.embed_in_layer.grads[0])
        self.grads.append(self.embed_dot_layer.grads[0])

        self.default_params_pickle_name = 'skipgram_params.p'

    def predict(self, context):
        embed_context = self.embed_in_layer.forward(context)
        score = self.embed_dot_layer.forward(py.arange(self.vocab_size), embed_context)
        return py.argmax(score, axis=1)

    def forward(self, context, true_targets):
        embed_context = self.embed_in_layer.forward(context)

        samples, sample_labels = self.negative_sampler.get_mixed_samples_and_labels(self.sample_size, true_targets)
        sample_scores = self.embed_dot_layer.forward(samples, embed_context)

        sample_loss = self.loss_layer.forward(sample_scores, sample_labels)
        avg_loss = py.average(sample_loss, axis=1)
        return avg_loss

    def backward(self, dout):
        davg_loss = dout.reshape(-1, 1)
        dsample_loss = py.repeat(davg_loss, self.sample_size, axis=1) / self.sample_size
        dscore = self.loss_layer.backward(dsample_loss)
        dembed_context = self.embed_dot_layer.backward(dscore)
        self.embed_in_layer.backward(dembed_context)
