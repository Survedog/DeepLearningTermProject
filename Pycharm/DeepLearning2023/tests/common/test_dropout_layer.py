import unittest
from common.dropout_layer import *
from common.utils import py


class DropOutLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        # 1
        dropout_rate = 0.0
        layer = DropOutLayer(dropout_rate)
        x = py.ones(10, dtype='f')

        fwd_result = layer.forward(x)
        self.assertTrue(py.array_equal(fwd_result, x))

        dout = py.ones_like(x)
        bwd_result = layer.backward(dout)
        self.assertTrue(py.array_equal(bwd_result, dout))

        # 2
        dropout_rate = 1.0
        layer = DropOutLayer(dropout_rate)
        x = py.ones((2, 5), dtype='f')

        fwd_result = layer.forward(x)
        self.assertTrue(py.allclose(fwd_result, py.zeros_like(x)))

        dout = py.ones_like(x)
        bwd_result = layer.backward(dout)
        self.assertTrue(py.allclose(bwd_result, py.zeros_like(dout)))

        # 3
        dropout_rate = 0.5
        layer = DropOutLayer(dropout_rate)
        x = py.ones((2, 2, 5), dtype='f')
        dout = py.ones_like(x)

        fwd_result = layer.forward(x)
        bwd_result = layer.backward(dout)
        self.assertTrue(py.allclose(fwd_result, bwd_result))