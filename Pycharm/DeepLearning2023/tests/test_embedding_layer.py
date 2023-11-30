import unittest
from embedding_layer import *


class EmbeddingLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        word_ids = [1, 2, 1]
        weight = np.array([[0, 1, 2, 3, 4],
                           [5, 6, 7, 8, 9],
                           [4, 3, 2, 1, 0],
                           [9, 8, 7, 6, 5]])
        embed_layer = EmbeddingLayer(weight)

        embedded = embed_layer.forward(word_ids)
        self.assertTrue(np.array_equal(embedded, weight[word_ids]))

        dout = np.array([[0, 1, 2, 3, 4],
                         [5, 6, 7, 8, 9],
                         [9, 8, 7, 6, 5]])
        dWeight = embed_layer.backward(dout)

        self.assertTrue(np.array_equal(dWeight, np.array([[0, 0, 0, 0, 0],
                                                          [9, 9, 9, 9, 9],
                                                          [5, 6, 7, 8, 9],
                                                          [0, 0, 0, 0, 0]])))


class EmbeddingDotLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        word_ids = [1, 2, 1]
        weight = np.array([[0, 1, 2, 3, 4],
                           [5, 6, 7, 8, 9],
                           [4, 3, 2, 1, 0],
                           [9, 8, 7, 6, 5]])
        h = np.array([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9],
                      [9, 8, 7, 6, 5]])

        embed_dot_layer = EmbeddingDotLayer(weight)
        score = embed_dot_layer.forward(word_ids, h)

        # dot result
        # [0, 6, 14, 24, 36],
        # [20, 18, 14, 8, 0],
        # [45, 48, 49, 48, 45]
        self.assertTrue(np.array_equal(score, np.array([80, 60, 235])))

        dout = np.array([0, 1, 2])
        dh = embed_dot_layer.backward(dout)

        self.assertTrue(np.array_equal(dh, np.array([[0, 0, 0, 0, 0],
                                                     [4, 3, 2, 1, 0],
                                                     [10, 12, 14, 16, 18]])))
