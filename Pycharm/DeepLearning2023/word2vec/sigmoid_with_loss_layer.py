from common.base_layer import LayerBase
from common.utils import py, sigmoid, get_class_cross_entropy


class SigmoidWithLossLayer(LayerBase):

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x, t):
        y = sigmoid(x)
        loss = get_class_cross_entropy(py.stack((1-y, y), axis=2), t)
        self.cache = (y, t)
        return loss

    def backward(self, dout):
        y, t = self.cache
        dx = (y - t) * dout
        return dx
