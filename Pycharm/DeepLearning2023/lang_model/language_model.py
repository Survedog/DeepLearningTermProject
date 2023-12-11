from common.base_layer import LayerBase
from common.loss_layers import SoftmaxWithLossLayer
from common.affine_layer import AffineLayer
from lang_model.time_embedding_layer import TimeEmbeddingLayer
from lang_model.time_rnn_layers import TimeLSTMLayer
from common.utils import py


class LanguageModel(LayerBase):

    def __init__(self, vocab_size, wordvec_size, hidden_size, embed_weight=None):
        super().__init__()

        if embed_weight is None:
            embed_weight = py.random.randn(vocab_size, wordvec_size, dtype='f') / 100
            load_embed_weight = False
        else:
            load_embed_weight = True

        lstm_weight_x = py.random.randn(wordvec_size, 4 * hidden_size, dtype='f') / py.sqrt(wordvec_size)
        lstm_weight_h = py.random.randn(hidden_size, 4 * hidden_size, dtype='f') / py.sqrt(hidden_size)
        lstm_bias = py.zeros(4 * hidden_size, dtype='f')
        affine_weight = py.random.randn(hidden_size, vocab_size, dtype='f') / py.sqrt(hidden_size)
        affine_bias = py.zeros(vocab_size, dtype='f')

        self.layers = [TimeEmbeddingLayer(embed_weight),
                       TimeLSTMLayer(lstm_weight_x, lstm_weight_h, lstm_bias),
                       AffineLayer(affine_weight, affine_bias)]
        self.loss_layer = SoftmaxWithLossLayer()

        save_grad_from = 1 if load_embed_weight else 0
        for i in range(save_grad_from, len(self.layers)):
            self.params += self.layers[i].params
            self.grads += self.layers[i].grads

        self.default_params_pickle_name = 'lm_params.p'

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        if xs.ndim == 1:
            xs = xs.reshape(1, -1)

        xs = self.predict(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.layers[1].reset_state()
