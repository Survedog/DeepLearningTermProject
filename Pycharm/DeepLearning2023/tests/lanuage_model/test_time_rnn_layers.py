import unittest
import traceback
from lang_model.time_rnn_layers import *


class TimeLSTMLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        batch_size = 10
        time_size = 10
        hidden_size = 5
        vocab_size = 20

        xs = py.random.randnint(0, vocab_size, (batch_size, time_size, vocab_size))
        dhs = py.ones((batch_size, time_size, hidden_size))

        weight_x = py.random.randn(vocab_size, hidden_size * 4)
        weight_h = py.random.randn(hidden_size, hidden_size * 4)
        bias = py.random.randn(batch_size, hidden_size * 4)

        try:
            layer = TimeLSTMLayer(weight_x, weight_h, bias)
            hs = layer.forward(xs)
            self.assertEqual(hs.shape, dhs.shape)

            dxs = layer.backward(dhs)
            self.assertEqual(xs.shape, dxs.shape)
        except:
            error_msg = traceback.format_exc()
            self.fail(msg=error_msg)
