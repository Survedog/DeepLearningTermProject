from common.base_layer import LayerBase
from lang_model.lstm_layer import LSTMLayer
from common.utils import py

# todo: LSTM 이외의 다른 RNN 모델에도 대응 가능하도록 수정
class TimeLSTMLayer(LayerBase):

    def __init__(self, weight_x, weight_h, bias, stateful=False):
        super().__init__()
        self.params += [weight_x, weight_h, bias]
        self.grads += [py.zeros_like(weight_x), py.zeros_like(weight_h), py.zeros_like(bias)]

        self.stateful = stateful

        self.h = None
        self.c = None
        self.dh = None
        self.layers = None

    def forward(self, xs):
        if xs.ndim < 3:
            xs = xs.reshape((1,) + xs.shape)

        batch_size, time_size, _ = xs.shape
        hidden_size = self.params[1].shape[0]

        hs = py.empty((batch_size, time_size, hidden_size))

        if not self.stateful or self.h is None:  # 상태를 이어 받지 않을 경우 새로 상태를 생성한다.
            self.h = py.zeros((batch_size, hidden_size))
            self.c = py.zeros((batch_size, hidden_size))

        self.layers = []
        for t in range(time_size):
            layer = LSTMLayer(*self.params)
            self.layers.append(layer)

            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

        return hs

    # todo: 기울기 소실/폭발 해결
    def backward(self, dhs):
        """
        :param dhs: 깊이 방향으로 출력된 h들의 열 hs에 대한 손실의 미분
        :return: dxs: 입력된 형태소 열 xs에 대한 손실의 미분
        """
        if dhs.ndim < 3:
            dhs = dhs.reshape((1,) + dhs.shape)

        batch_size, time_size, hidden_size = dhs.shape
        vocab_size = self.params[0].shape[0]

        # 마지막 RNN 모델은 깊이 방향으로만 손실에 관여하므로 0을 넣는다.
        dh = py.zeros((batch_size, hidden_size))
        dc = py.zeros((batch_size, hidden_size))

        dxs = py.empty((batch_size, time_size, vocab_size))

        grads = [py.zeros_like(self.params[0]), py.zeros_like(self.params[1]), py.zeros_like(self.params[2])]
        for t in reversed(range(time_size)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dh + dhs[:, t, :], dc)

            dxs[:, t, :] = dx
            for i in range(len(layer.grads)):
                grads[i] += layer.grads[i]

        for i in range(len(grads)):
            self.grads[i][...] = grads[i]

        return dxs

    def set_state(self, h, c):
        self.h = h
        self.c = c

    def reset_state(self):
        self.h = None
        self.c = None
