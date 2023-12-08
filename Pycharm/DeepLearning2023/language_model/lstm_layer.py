from base_layer import LayerBase
from utils import py, sigmoid


class LSTMLayer(LayerBase):

    def __init__(self, weight_x, weight_h, bias):
        super().__init__()
        self.params += [weight_x, weight_h, bias]
        self.grads += [py.zeros_like(weight_x), py.zeros_like(weight_h), py.zeros_like(bias)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        weight_x, weight_h, bias = self.params
        H = h_prev.shape[1]  # hidden_size

        affine = py.matmul(weight_x, x) + py.matmul(weight_h, h_prev) + bias
        f, g, i, o = affine[:, :H], affine[:, H:2*H], affine[:, 2*H:3*H], affine[:, 3*H:4*H]

        # todo: affine에 sigmoid 한번에 계산 시 시간 단축되는지 확인
        f = sigmoid(f)
        i = sigmoid(i)
        o = sigmoid(o)
        g = py.tanh(g)

        c = c_prev * f + g * i
        tanh_c = py.tanh(c)
        h = tanh_c * o
        self.cache = (x, h_prev, c_prev, tanh_c, f, g, i, o)
        return h, c

    def backward(self, dh, dc):
        x, h_prev, c_prev, tanh_c, f, g, i, o = self.cache

        dtanh_c = dh * o
        ds = dc + dtanh_c * (1 - py.square(tanh_c))

        dc_prev = ds * f
        do = dh * tanh_c
        df = ds * c_prev
        dg = ds * i
        di = ds * g
        daffine = py.hstack((df*(1-df), 1 - py.square(dg), di*(1-di), do*(1-do)))

        weight_x, weight_h, _ = self.params
        dweight_x, dweight_h, dbias = self.grads

        dweight_x[...] = py.matmul(x.T, daffine)
        dweight_h[...] = py.matmul(h_prev.T, daffine)
        dbias[...] = daffine

        dh_prev = py.matmul(daffine, weight_h.T)
        dx = py.matmul(daffine, weight_x.T)

        return dx, dh_prev, dc_prev
