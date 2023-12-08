from common.base_layer import LayerBase
from common.loss_layers import SoftmaxWithLossLayer
from common.affine_layer import AffineLayer
from word2vec.embedding_layer import EmbeddingLayer
from time_rnn_layers import TimeLSTMLayer
from common.utils import py


class LanguageModel(LayerBase):

    def __init__(self, vocab_size, wordvec_size, hidden_size, embed_weight=None):
        super().__init__()

        load_embed_weight = embed_weight is not None

        lstm_weight_x = py.random.rand(wordvec_size, 4 * hidden_size)
        lstm_weight_h = py.random.rand(hidden_size, 4 * hidden_size)
        lstm_bias = py.zeros(4 * hidden_size)
        affine_weight = py.random.rand(hidden_size, vocab_size)
        affine_bias = py.zeros(vocab_size)

        self.layers = [EmbeddingLayer(embed_weight),
                       TimeLSTMLayer(lstm_weight_x, lstm_weight_h, lstm_bias),
                       AffineLayer(affine_weight, affine_bias)]
        self.loss_layer = SoftmaxWithLossLayer()

        save_grad_from = 1 if load_embed_weight else 0
        for i in range(save_grad_from, len(self.layers)):
            self.params += self.layers[i].params
            self.grads += self.layers[i].grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        xs = self.predict(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
