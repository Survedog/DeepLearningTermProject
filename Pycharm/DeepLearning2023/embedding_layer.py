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
        dW = self.grads[0]
        dW[...] = np.zeros_like(0)

        for dout_idx, word_id, in enumerate(self.word_ids):
            dW[word_id] += dout[dout_idx]
        return dW
