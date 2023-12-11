from common.base_layer import LayerBase
from word2vec.embedding_layer import EmbeddingLayer
from common.utils import py


class TimeEmbeddingLayer(LayerBase):

    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.params.append(weight)
        self.grads.append(py.zeros_like(weight))
        self.layers = None

    def forward(self, xs):
        batch_size, time_size = xs.shape
        wordvec_size = self.weight.shape[-1]

        self.layers = []
        out = py.empty((batch_size, time_size, wordvec_size), dtype='f')

        for time in range(time_size):
            embed_layer = EmbeddingLayer(self.weight)
            out[:, time, :] = embed_layer.forward(xs[:, time])
            self.layers.append(embed_layer)

        return out

    def backward(self, dout):
        batch_size, time_size, wordvec_size = dout.shape
        self.grads[0][...] = 0

        for time in reversed(range(time_size)):
            layer = self.layers[time]
            layer.backward(dout[:, time, :])
            self.grads[0] += layer.grads[0]
