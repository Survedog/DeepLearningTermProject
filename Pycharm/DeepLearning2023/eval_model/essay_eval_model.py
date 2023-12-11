from common.base_layer import LayerBase
from common.affine_layer import AffineLayer
from word2vec.embedding_layer import EmbeddingLayer


class EssayEvalModel(LayerBase):

    def __init__(self, embed_weight, rnn_model):
        super().__init__()
        self.embed_in_layer = EmbeddingLayer(embed_weight)
        self.rnn_model = rnn_model

    def predict(self, x):
        pass

    def forward(self, x, t):
        pass

    def backward(self):
        pass
