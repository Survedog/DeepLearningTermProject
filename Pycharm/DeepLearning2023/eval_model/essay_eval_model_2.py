from common.base_layer import LayerBase
from common.affine_layer import AffineLayer
from common.utils import py, load_data, save_data
from common.loss_layers import SSELossLayer
from common.dropout_layer import DropOutLayer
from lang_model.time_rnn_layers import TimeLSTMLayer
from lang_model.time_embedding_layer import TimeEmbeddingLayer
import math


class EssayEvalModel2(LayerBase):

    def __init__(self, vocab_size, wordvec_size=100, lstm1_hidden_size=30, lstm2_hidden_size=10, time_size=50, dropout_rate=0.3, embed_weight=None):
        super().__init__()
        self.cache = None

        self.time_size = time_size
        self.lstm1_hidden_size = lstm1_hidden_size
        self.lstm2_hidden_size = lstm2_hidden_size

        randn = py.random.randn
        if embed_weight is None:
            embed_weight = randn(vocab_size, wordvec_size, dtype='f') / 100
            self.default_params_pickle_name = 'essay_eval_model_2_params_w_embed_weight.p'
            fit_from = 0
        else:
            self.default_params_pickle_name = 'essay_eval_model_2_params.p'
            fit_from = 1

        lstm1_weight_x = randn(wordvec_size, 4 * lstm1_hidden_size, dtype='f') / py.sqrt(wordvec_size)
        lstm1_weight_h = randn(lstm1_hidden_size, 4 * lstm1_hidden_size, dtype='f') / py.sqrt(lstm1_hidden_size)
        lstm1_bias = py.zeros(4 * lstm1_hidden_size, dtype='f')

        lstm2_weight_x = randn(lstm1_hidden_size, 4 * lstm2_hidden_size, dtype='f') / py.sqrt(lstm1_hidden_size)
        lstm2_weight_h = randn(lstm2_hidden_size, 4 * lstm2_hidden_size, dtype='f') / py.sqrt(lstm2_hidden_size)
        lstm2_bias = py.zeros(4 * lstm2_hidden_size, dtype='f')

        measures_size = 12
        affine_input_size = lstm2_hidden_size + measures_size
        criteria_amount = 11

        affine_weight = randn(affine_input_size, criteria_amount * 3, dtype='f')
        affine_bias = randn(criteria_amount * 3, dtype='f')

        self.dropout_layers, self.time_lstm_layers = [], []

        self.time_embedding_layer = TimeEmbeddingLayer(embed_weight)
        self.dropout_layers.append(DropOutLayer(dropout_rate))
        self.time_lstm_layers.append(TimeLSTMLayer(lstm1_weight_x, lstm1_weight_h, lstm1_bias))
        self.dropout_layers.append(DropOutLayer(dropout_rate))
        self.time_lstm_layers.append(TimeLSTMLayer(lstm2_weight_x, lstm2_weight_h, lstm2_bias))
        self.dropout_layers.append(DropOutLayer(dropout_rate))
        self.affine_layer = AffineLayer(affine_weight, affine_bias)

        self.loss_layer = SSELossLayer()
        self.layers = [self.time_embedding_layer,
                       self.dropout_layers[0],
                       self.time_lstm_layers[0],
                       self.dropout_layers[1],
                       self.time_lstm_layers[1],
                       self.dropout_layers[2],
                       self.affine_layer]

        for i in range(fit_from, len(self.layers)):
            layer = self.layers[i]
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x, train_flag=True):
        xs, measures = x

        embed_xs = self.time_embedding_layer.forward(xs)
        embed_xs = self.dropout_layers[0].forward(embed_xs, train_flag)

        self.time_lstm_layers[0].reset_state()
        hs1 = self.time_lstm_layers[0].forward(embed_xs)
        hs1 = self.dropout_layers[1].forward(hs1, train_flag)

        self.time_lstm_layers[1].reset_state()
        hs2 = self.time_lstm_layers[1].forward(hs1)
        hs2 = self.dropout_layers[2].forward(hs2, train_flag)
        rhs = hs2.reshape(-1, self.lstm2_hidden_size)

        measures = py.hstack((measures[0], measures[1], measures[2]))
        measures_repeated = measures[py.newaxis].repeat(rhs.shape[0], axis=0)
        x = py.hstack((rhs, measures_repeated))
        scores = self.affine_layer.forward(x).mean(axis=0)

        self.cache = (hs1.shape, hs2.shape, rhs.shape)
        return scores

    def forward(self, x, t, train_flag=True):
        scores = self.predict(x, train_flag)
        loss = self.loss_layer.forward(scores, t)
        return loss

    def backward(self, dout=1):
        hs1_shape, hs2_shape, rhs_shape = self.cache

        dscores = self.loss_layer.backward(dout)
        dscores = dscores[py.newaxis].repeat(rhs_shape[0], axis=0) / rhs_shape[0]

        drhs = self.affine_layer.backward(dscores)[:, :rhs_shape[-1]]
        dhs2 = drhs.reshape(hs2_shape)
        dhs2 = self.dropout_layers[2].backward(dhs2)

        dhs1 = self.time_lstm_layers[1].backward(dhs2)
        dhs1 = self.dropout_layers[1].backward(dhs1)

        dembed_xs = self.time_lstm_layers[0].backward(dhs1)
        dembed_xs = self.dropout_layers[0].backward(dembed_xs)
        self.time_embedding_layer.backward(dembed_xs)

    def reset_state(self):
        for layer in self.time_lstm_layers:
            layer.reset_state()

    @classmethod
    def get_x_t_list_from_processed_data(cls, data_list, time_size, load_pickle=True, save_pickle=False, pickle_name='eval_model_args.p'):
        if load_pickle:
            data = load_data(pickle_name)
            if data is not None:
                return data

        x_list, t_list = [], []

        for data in data_list:
            try:
                xs = py.array(sum(data['paragraph'], []))
            except TypeError:
                continue  # paragraph를 온전히 가져오지 못한 데이터는 생략한다.

            split_amount = math.ceil(len(xs) / time_size)
            xs = py.resize(xs, (1, split_amount * time_size)).reshape(-1, time_size)  # time size에 맞는 크기가 될 때까지 내용을 반복시킨다.

            exp_weight, org_weight, cont_weight = data['weight']['exp'], data['weight']['org'], data['weight']['cont']
            # 대분류 가중치를 하위의 소분류 가중치에 곱한다.
            measures = [exp_weight['exp'] * py.array([exp_weight['exp_grammar'], exp_weight['exp_vocab'], exp_weight['exp_style'], data['corr_count'] * exp_weight['exp_grammar']]),  # 문법 점수 계산에 교정 횟수를 포함
                        org_weight['org'] * py.array([org_weight['org_paragraph'], org_weight['org_essay'], org_weight['org_coherence'], org_weight['org_quantity']]),
                        cont_weight['con'] * py.array([cont_weight['con_clearance'], cont_weight['con_novelty'], cont_weight['con_prompt'], cont_weight['con_description']])]

            t = sum(data['score']['exp'], []) + sum(data['score']['org'], []) + sum(data['score']['cont'], [])
            t = py.array(t)
            t_list.append(t)

            x = (xs, measures)
            x_list.append(x)

        if save_pickle:
            save_data(pickle_name, (x_list, t_list))

        return x_list, t_list
