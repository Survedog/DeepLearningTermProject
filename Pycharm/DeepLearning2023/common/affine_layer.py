from common.base_layer import LayerBase
from common.utils import py


class AffineLayer(LayerBase):

    def __init__(self, weight, bias):
        super().__init__()
        self.params += [weight, bias]
        self.grads += [py.zeros_like(weight), py.zeros_like(bias)]
        self.x = None

    def forward(self, x):
        '''
        :param x: 2차원 입력
        :return out: 다음 계층의 노드 값들
        '''
        self.x = x
        if x.ndim >= 3:
            x = x.reshape(-1, x.shape[-1])

        weight, bias = self.params
        out = py.matmul(x, weight) + bias

        if self.x.ndim >= 3:
            shape = self.x.shape[:-1] + (-1,)
            out = out.reshape(shape)
        return out

    def backward(self, dout):
        x = self.x
        if dout.ndim >= 3:
            dout = dout.reshape(-1, dout.shape[-1])
            x = x.reshape(-1, x.shape[-1])

        weight = self.params[0]
        dweight, dbias = self.grads

        dweight[...] = py.matmul(x.T, dout)
        dbias[...] = py.sum(dout, axis=0)
        dx = py.matmul(dout, weight.T)

        if self.x.ndim >= 3:
            shape = self.x.shape[:-1] + (-1,)
            dx = dx.reshape(shape)
        return dx

