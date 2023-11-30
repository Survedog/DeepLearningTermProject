import unittest

import numpy as np

from sigmoid_with_loss_layer import *
from utils import get_class_cross_entropy, sigmoid


class SigmoidWithLossLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        x = np.array([0.2, 0.1, 0.7])
        t = np.array([1, 0, 1])

        layer = SigmoidWithLoss()
        loss = layer.forward(x, t)

        y = sigmoid(x)
        y = np.array([[1 - y[0], y[0]],
                      [1 - y[1], y[1]],
                      [1 - y[2], y[2]]])
        correct_loss = get_class_cross_entropy(y, t)
        self.assertTrue(np.array_equal(loss, correct_loss))

        dout = np.array([1, 1, 1])
        dx = layer.backward(dout)
        correct_dx = sigmoid(x) - t
        self.assertTrue(np.array_equal(dx, correct_dx))


