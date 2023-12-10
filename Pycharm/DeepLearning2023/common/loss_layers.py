from common.base_layer import LayerBase
from common.utils import py, sigmoid, softmax, get_class_cross_entropy, get_one_hot_encoding


class SigmoidWithLossLayer(LayerBase):

    def __init__(self):
        super().__init__()
        self.cache = None

    # todo: 평균 loss만 반환하도록 변경
    def forward(self, x, t):
        y = sigmoid(x)
        loss = get_class_cross_entropy(py.stack((1-y, y), axis=2), t)
        self.cache = (y, t)
        return loss

    def backward(self, dout=1):
        y, t = self.cache
        dx = (y - t) * dout
        return dx


class SoftmaxWithLossLayer(LayerBase):

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x, t):
        y = softmax(x)
        loss = get_class_cross_entropy(y, t)
        self.cache = (y, t)
        return py.average(loss)

    def backward(self, dout=1):
        y, t = self.cache
        t = get_one_hot_encoding(t, y.shape[1])
        dx = (y - t) * dout
        return dx
