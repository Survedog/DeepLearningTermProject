import unittest
from utils import py
from word2vec.embedding_layer import *


class EmbeddingLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        # 1차원 입력
        word_ids = [1, 2, 1]
        weight = py.array([[0, 1, 2, 3, 4],
                           [5, 6, 7, 8, 9],
                           [4, 3, 2, 1, 0],
                           [9, 8, 7, 6, 5]])
        embed_layer = EmbeddingLayer(weight)

        embedded = embed_layer.forward(word_ids)
        self.assertTrue(py.array_equal(embedded, weight[word_ids]))

        dout = py.array([[0, 1, 2, 3, 4],
                         [5, 6, 7, 8, 9],
                         [9, 8, 7, 6, 5]])
        dWeight = embed_layer.backward(dout)

        self.assertTrue(py.array_equal(dWeight, py.array([[0, 0, 0, 0, 0],
                                                          [9, 9, 9, 9, 9],
                                                          [5, 6, 7, 8, 9],
                                                          [0, 0, 0, 0, 0]])))

        # 2차원 입력 (배치 처리)
        word_ids = [[1, 2],
                    [0, 0]]
        weight = py.array([[0, 1, 2, 3, 4],
                           [5, 6, 7, 8, 9],
                           [4, 3, 2, 1, 0]])

        embed_layer = EmbeddingLayer(weight)
        embedded = embed_layer.forward(word_ids)
        self.assertTrue(py.array_equal(embedded, weight[word_ids]))

        dout = py.array([[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                         [[5, 6, 7, 8, 9], [9, 8, 7, 6, 5]]])
        dWeight = embed_layer.backward(dout)
        self.assertTrue(
            py.array_equal(dWeight, py.array([dout[1][0] + dout[1][1],
                                              dout[0][0],
                                              dout[0][1]])))


class EmbeddingDotLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        word_ids = [[1, 2],
                    [0, 0]]
        weight = py.array([[9, 8, 7, 6, 5],
                           [0, 1, 2, 3, 4],
                           [4, 3, 2, 1, 0]])
        h = py.array([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9]])

        embed_dot_layer = EmbeddingDotLayer(weight)
        score = embed_dot_layer.forward(word_ids, h)

        # embedded: [[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]],
        #            [[9, 8, 7, 6, 5], [9, 8, 7, 6, 5]]]
        # dotted: [[[0, 1, 4, 9, 16], [0, 3, 4, 3, 0]],
        #          [[45, 48, 49, 48, 45], [45, 48, 49, 48, 45]]]
        correct_score = py.array([[30, 10],
                                  [235, 235]])

        self.assertTrue(py.array_equal(score, correct_score))

        dout = py.array([[0, 1],
                         [2, 1]])
        dh = embed_dot_layer.backward(dout)

        # dh은 dout와 embedded의 곱
        correct_dh = py.array([[4, 3, 2, 1, 0],
                               [27, 24, 21, 18, 15]])
        self.assertTrue(py.array_equal(dh, correct_dh))

        dWeight = embed_dot_layer.grads[0]
        correct_dWeight = py.array([[15, 18, 21, 24, 27],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 2, 3, 4]])
        self.assertTrue(py.array_equal(dWeight, correct_dWeight))
