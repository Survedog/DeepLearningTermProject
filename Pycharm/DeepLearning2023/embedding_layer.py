from base_layer import LayerBase
from config import Config

if Config.USE_GPU:
    import cupy as np
else:
    import numpy as np


class EmbeddingLayer(LayerBase):

    def __init__(self, weight):
        super().__init__()
        self.params.append(weight)
        self.grads.append(np.zeros_like(weight))
        self.word_ids = None
        pass

    def forward(self, word_ids):
        self.word_ids = word_ids
        weight = self.params[0]
        return weight[word_ids]

    def backward(self, dout):
        dWeight = self.grads[0]
        dWeight[...] = np.zeros_like(0)

        for dout_idx, word_id, in enumerate(self.word_ids):
            dWeight[word_id] += dout[dout_idx]
        return dWeight


class EmbeddingDotLayer(LayerBase):

    def __init__(self, weight):
        super().__init__()
        self.params.append(weight)
        self.grads.append(np.zeros_like(weight))
        self.embed_layer = EmbeddingLayer(weight)
        self.cache = None

    def forward(self, word_ids, h):
        embedded = self.embed_layer.forward(word_ids)
        dotted = h * embedded
        self.cache = (h, embedded)
        return np.sum(dotted, axis=1)

    def backward(self, dout):
        h, embedded = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dEmbed = h * dout
        self.embed_layer.backward(dEmbed)
        dh = embedded * dout
        return dh
