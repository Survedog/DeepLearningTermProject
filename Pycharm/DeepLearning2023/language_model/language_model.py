from common.base_layer import LayerBase
from common.loss_layers import SoftmaxWithLossLayer
from common.affine_layer import AffineLayer
from word2vec.embedding_layer import EmbeddingLayer
from time_rnn_layers import TimeLSTMLayer
from common.utils import py


class LanguageModel(LayerBase):

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__()

        # todo: wordvec에 word2vec의 가중치 사용할 수 있는지 확인
        embed_weight = py.random.rand(vocab_size, wordvec_size)
        lstm_weight_x = py.random.rand(wordvec_size, 4 * hidden_size)
        lstm_weight_h = py.random.rand(hidden_size, 4 * hidden_size)
        lstm_bias = py.zeros(4 * hidden_size)
        affine_weight = py.random.rand(hidden_size, vocab_size)
        affine_bias = py.zeros(vocab_size)

        self.layers = [EmbeddingLayer(embed_weight),
                       TimeLSTMLayer(lstm_weight_x, lstm_weight_h, lstm_bias),
                       AffineLayer(affine_weight, affine_bias)]
        self.loss_layer = SoftmaxWithLossLayer()

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

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
