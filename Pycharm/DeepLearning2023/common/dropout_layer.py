from common.base_layer import LayerBase
from common.utils import py


class DropOutLayer(LayerBase):

    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.out_mask = None

    def forward(self, x, train_flag=True):
        if train_flag:
            self.out_mask = py.random.rand(*x.shape) < self.dropout_rate
            x[self.out_mask] = 0
        return x

    def backward(self, dout):
        dout[self.out_mask] = 0
        return dout
