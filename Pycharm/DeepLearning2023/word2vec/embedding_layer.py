from common.base_layer import LayerBase
from common.utils import py


class EmbeddingLayer(LayerBase):

    def __init__(self, weight):
        super().__init__()
        self.params.append(weight)
        self.grads.append(py.zeros_like(weight))
        self.word_ids = None

    def forward(self, word_ids):
        self.word_ids = word_ids
        weight = self.params[0]
        return weight[word_ids]

    def backward(self, dout):
        dWeight = self.grads[0]
        dWeight[...] = 0

        py.add.at(dWeight, self.word_ids, dout)


class EmbeddingDotLayer(LayerBase):

    def __init__(self, weight):
        super().__init__()
        self.embed_layer = EmbeddingLayer(weight)
        self.params.append(weight)
        self.grads += self.embed_layer.grads
        self.cache = None

    def forward(self, word_ids, h):
        """
        :param word_ids: 단어 id 샘플 리스트 (N x S)
        :param h: 내적할 은닉층 값 (N x H)
        :return: 은닉층과 각 샘플의 내적 값 (N x S)
        """
        embedded = self.embed_layer.forward(word_ids)
        h = py.expand_dims(h, axis=1)
        dotted = h * embedded
        self.cache = (h, embedded)
        return py.sum(dotted, axis=2)

    def backward(self, dout):
        """
        :param dout: 다음 레이어에서 전달한 각 샘플의 역전파 값 배치 (N x S)
        :return: 은닉층에 대한 최종 값의 미분 dh (N x H)
        """
        h, embedded = self.cache

        # (N, 1, H) * (N, S, 1) -> (N, S, H)
        dEmbed = h * py.expand_dims(dout, axis=2)
        self.embed_layer.backward(dEmbed)

        # (N, S, H) * (N, S, 1) -> (N, S, H)
        dh = embedded * py.expand_dims(dout, axis=2)

        # (N, S, H) -> (N, H)
        dh = py.sum(dh, axis=1)
        return dh
