from common.base_layer import LayerBase
from word2vec.embedding_layer import EmbeddingLayer, EmbeddingDotLayer
from common.loss_layers import SigmoidWithLossLayer
from word2vec.negative_sampler import NegativeSampler
from common.utils import py


class SkipgramModel(LayerBase):

    def __init__(self, corpus, vocab_size, wordvec_size, sample_size, weight_in, weight_out_list):
        super().__init__()
        self.vocab_size = vocab_size
        self.wordvec_size = wordvec_size
        self.sample_size = sample_size
        self.target_size = len(weight_out_list)

        distribution = py.bincount(corpus, minlength=vocab_size) / len(corpus)
        self.negative_sampler = NegativeSampler(py.unique(corpus), distribution)

        self.loss_layer = SigmoidWithLossLayer()
        self.embed_in_layer = EmbeddingLayer(weight_in)
        self.embed_dot_layers = []
        for weight_out in weight_out_list:
            self.embed_dot_layers.append(EmbeddingDotLayer(weight_out))

        self.params.append(weight_in)
        self.grads.append(self.embed_in_layer.grads[0])

        for i in range(len(weight_out_list)):
            self.params.append(weight_out_list[i])
            self.grads.append(self.embed_dot_layers[i].grads[0])

        self.default_params_pickle_name = 'skipgram_params.p'

    def predict(self, context):
        batch_size = context.shape[0]

        embed_context = self.embed_in_layer.forward(context)
        scores = py.empty((batch_size, self.target_size, self.vocab_size), dtype='f')
        for i in range(self.target_size):
            scores[:, i, :] = self.embed_dot_layers[i].forward(py.arange(self.vocab_size), embed_context)

        return py.argmax(scores, axis=2)

    def forward(self, context, true_targets):
        batch_size = context.shape[0]

        embed_context = self.embed_in_layer.forward(context)
        samples, sample_labels = self.negative_sampler.get_mixed_samples_and_labels(self.sample_size, true_targets)

        sample_loss = py.zeros((batch_size, self.sample_size), dtype='f')
        for i in range(self.target_size):
            sample_scores = self.embed_dot_layers[i].forward(samples, embed_context)
            sample_loss += self.loss_layer.forward(sample_scores, sample_labels[i])

        sample_loss /= self.target_size
        avg_loss = py.average(sample_loss, axis=1)
        return avg_loss

    def backward(self, dout):
        davg_loss = dout.reshape(-1, 1)
        dsample_loss = py.repeat(davg_loss, self.sample_size, axis=1) / self.sample_size
        dscore = self.loss_layer.backward(dsample_loss / self.target_size)

        dembed_context = 0
        for embed_dot_layer in self.embed_dot_layers:
            dembed_context += embed_dot_layer.backward(dscore)
        self.embed_in_layer.backward(dembed_context)
