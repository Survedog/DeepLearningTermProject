import unittest
from embedding_layer import *
from utils import get_one_hot_encoding

class EmbeddingLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        weight = np.random.rand(4, 5)
        embed_layer = EmbeddingLayer(weight)

        word_ids = [1, 2, 1]
        embedded = embed_layer.forward(word_ids)
        self.assertTrue(np.array_equal(embedded, weight[word_ids]))

        dout = np.random.rand(embedded.shape[0], embedded.shape[1])
        dW = embed_layer.backward(dout)

        answer = np.zeros_like(weight)
        for idx, word_id in enumerate(word_ids):
            answer[word_id] += dout[idx]
        self.assertTrue(np.array_equal(dW, answer))



