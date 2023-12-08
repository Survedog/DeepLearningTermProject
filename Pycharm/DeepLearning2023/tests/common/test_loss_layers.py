import unittest

from common.loss_layers import *
from common.utils import py, get_class_cross_entropy, sigmoid


class SigmoidWithLossLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        # 2차원 입력
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


class SoftmaxWithLossLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        layer = SoftmaxWithLossLayer()

        # 2차원 입력
        x = py.array([[0.1, 0.05, 0.7, 0.15],
                      [0.5, 0.3, 0.1, 0.1]])
        t = py.array([2, 0])

        loss = layer.forward(x, t)

        y = softmax(x)
        correct_loss = -py.log(py.array([y[0, 2], y[1, 0]]))
        self.assertTrue(py.allclose(loss, correct_loss))

        dout = py.array([[1, 0.5, 1, 0.3],
                         [1, 0.4, 0.2, 0.1]])
        dx = layer.backward(dout)

        t = get_one_hot_encoding(t, y.shape[1])
        correct_dx = (y - t) * dout
        self.assertTrue(py.allclose(dx, correct_dx))


