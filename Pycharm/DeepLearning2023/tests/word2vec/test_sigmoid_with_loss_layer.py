import unittest

from word2vec.sigmoid_with_loss_layer import *
from common.utils import py, get_class_cross_entropy, sigmoid


class SigmoidWithLossLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        # 2차원 입력 (배치 처리)
        x = py.array([[0.2, 0.1, 0.7],
                      [0.5, 0.4, 0.1]])
        t = py.array([[1, 0, 1],
                      [0, 1, 0]])

        layer = SigmoidWithLossLayer()
        loss = layer.forward(x, t)

        y = sigmoid(x)
        y = py.array([[[1 - y[0][0], y[0][0]],
                       [1 - y[0][1], y[0][1]],
                       [1 - y[0][2], y[0][2]]],

                      [[1 - y[1][0], y[1][0]],
                       [1 - y[1][1], y[1][1]],
                       [1 - y[1][2], y[1][2]]]])

        correct_loss = get_class_cross_entropy(y, t)
        self.assertTrue(py.array_equal(loss, correct_loss))

        dout = py.array([[1, 0.5, 1],
                         [0.3, 1, 0.2]])
        dx = layer.backward(dout)
        correct_dx = (sigmoid(x) - t) * dout
        self.assertTrue(py.array_equal(dx, correct_dx))
