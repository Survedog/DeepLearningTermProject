from common.base_layer import LayerBase
from common.utils import py


class AffineLayer(LayerBase):

    def __init__(self, weight, bias):
        super().__init__()
        self.params += [weight, bias]
        self.grads += [py.zeros_like(weight), py.zeros_like(bias)]
        self.cache = None

    def forward(self, x):
        '''
        :param x: 2차원 입력
        :return out: 다음 계층의 노드 값들
        '''
        # todo: 3차원 입력이 들어오면 2차원으로 변환하기
        weight, bias = self.params
        out = py.matmul(x, weight) + bias

        self.cache = x
        return out

    def backward(self, dout):
        x = self.cache
        weight = self.params[0]
        dweight, dbias = self.grads

        dx = py.matmul(dout, weight.T)
        dweight[...] = py.matmul(x.T, dout)
        dbias[...] = py.sum(dout, axis=0)

        return dx

