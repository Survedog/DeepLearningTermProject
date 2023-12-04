from base_layer import LayerBase
from utils import sigmoid, get_class_cross_entropy
from utils import np


class SigmoidWithLossLayer(LayerBase):

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x, t):
        y = sigmoid(x)
        loss = get_class_cross_entropy(np.stack((1-y, y), axis=2), t)
        self.cache = (y, t)
        return loss

    def backward(self, dout):
        y, t = self.cache
        #todo: batch size로 나눠야 하면 적용
        dx = (y - t) * dout
        return dx