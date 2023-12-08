import unittest
import traceback
from language_model.lstm_layer import LSTMLayer
from utils import py


class LSTMLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):

        batch_size = 10
        vocab_size = 20
        hidden_size = 5

        x = py.random.randint(0, vocab_size, (batch_size, vocab_size))
        h_prev = py.zeros((batch_size, hidden_size))
        c_prev = py.zeros((batch_size, hidden_size))

        dh = py.ones_like(h_prev)
        dc = py.ones_like(c_prev)

        weight_x = py.random.rand(vocab_size, hidden_size * 4)
        weight_h = py.random.rand(hidden_size, hidden_size * 4)
        bias = py.random.rand(batch_size, hidden_size * 4)

        try:
            layer = LSTMLayer(weight_x, weight_h, bias)
            h, c = layer.forward(x, h_prev, c_prev)
            self.assertEqual(h.shape, h_prev.shape)
            self.assertEqual(c.shape, c_prev.shape)

            dx, dh_prev, dc_prev = layer.backward(dh, dc)
            self.assertEqual(dh.shape, dh_prev.shape)
            self.assertEqual(dc.shape, dc_prev.shape)
        except:
            error_msg = traceback.format_exc()
            self.fail(msg=error_msg)
