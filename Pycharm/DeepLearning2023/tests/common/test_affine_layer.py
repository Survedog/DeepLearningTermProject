import unittest
from common.affine_layer import *
from common.utils import py


class AffineLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        weight = py.random.rand(4, 5)
        bias = py.random.rand(5)
        layer = AffineLayer(weight, bias)

        # 2차원 입력
        x = py.random.rand(10, 4)
        out = layer.forward(x)
        self.assertEqual(out.shape, (10, 5))

        dout = py.ones((10, 5))
        dx = layer.backward(dout)
        self.assertEqual(dx.shape, x.shape)

        # 3차원 입력
        x = py.random.rand(2, 5, 4)
        out = layer.forward(x)
        self.assertEqual(out.shape, (10, 5))

        dout = py.ones((10, 5))
        dx = layer.backward(dout)
        self.assertEqual(dx.shape, (10, 4))
