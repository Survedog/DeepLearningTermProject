from common.base_layer import LayerBase
from common.affine_layer import AffineLayer
from common.utils import py, load_data, save_data
from common.loss_layers import SSELossLayer
from lang_model.time_rnn_layers import TimeLSTMLayer
from lang_model.time_embedding_layer import TimeEmbeddingLayer
import math


class EssayEvalModel(LayerBase):

    def __init__(self, vocab_size, wordvec_size=100, lstm_hidden_size=30, time_size=50, embed_weight=None):
        super().__init__()
        self.cache = None

        self.time_size = time_size
        self.lstm_hidden_size = lstm_hidden_size

        randn = py.random.randn
        if embed_weight is None:
            embed_weight = randn(vocab_size, wordvec_size, dtype='f') / 100
            self.default_params_pickle_name = 'essay_eval_model_params_w_embed_weight.p'
            fit_from = 0
        else:
            self.default_params_pickle_name = 'essay_eval_model_params.p'
            fit_from = 1

        lstm_weight_x = py.random.randn(wordvec_size, 4 * lstm_hidden_size, dtype='f') / py.sqrt(wordvec_size)
        lstm_weight_h = py.random.randn(lstm_hidden_size, 4 * lstm_hidden_size, dtype='f') / py.sqrt(lstm_hidden_size)
        lstm_bias = py.zeros(4 * lstm_hidden_size, dtype='f')

        affine_input_size = time_size * lstm_hidden_size + 4
        exp_metrics_amount, org_metrics_amount, cont_metrics_amount = 3, 4, 4

        exp_affine_weight = randn(affine_input_size, exp_metrics_amount * 3, dtype='f')
        exp_affine_bias = randn(exp_metrics_amount * 3, dtype='f')
        org_affine_weight = randn(affine_input_size, org_metrics_amount * 3, dtype='f')
        org_affine_bias = randn(org_metrics_amount * 3, dtype='f')
        cont_affine_weight = randn(affine_input_size, cont_metrics_amount * 3, dtype='f')
        cont_affine_bias = randn(cont_metrics_amount * 3, dtype='f')

        self.time_embedding_layer = TimeEmbeddingLayer(embed_weight)
        self.time_lstm_layer = TimeLSTMLayer(lstm_weight_x, lstm_weight_h, lstm_bias)
        self.exp_affine_layer = AffineLayer(exp_affine_weight, exp_affine_bias)
        self.org_affine_layer = AffineLayer(org_affine_weight, org_affine_bias)
        self.cont_affine_layer = AffineLayer(cont_affine_weight, cont_affine_bias)

        self.loss_layer = SSELossLayer()
        self.layers = [self.time_embedding_layer, self.time_lstm_layer, self.exp_affine_layer, self.org_affine_layer, self.cont_affine_layer]

        for i in range(fit_from, len(self.layers)):
            layer = self.layers[i]
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        xs, score_metrics = x

        embed_xs = self.time_embedding_layer.forward(xs)
        self.time_lstm_layer.reset_state()
        hs = self.time_lstm_layer.forward(embed_xs)
        # todo: hs 값을 평균내볼까?
        rhs = hs.reshape(-1, self.time_size * self.lstm_hidden_size)

        metrics_repeated = []
        for i in range(len(score_metrics)):
            metrics_repeated.append(score_metrics[i][py.newaxis].repeat(rhs.shape[0], axis=0))

        exp_x = py.hstack((rhs, metrics_repeated[0]))
        org_x = py.hstack((rhs, metrics_repeated[1]))
        cont_x = py.hstack((rhs, metrics_repeated[2]))

        exp_scores = self.exp_affine_layer.forward(exp_x).mean(axis=0)
        org_scores = self.org_affine_layer.forward(org_x).mean(axis=0)
        cont_scores = self.cont_affine_layer.forward(cont_x).mean(axis=0)

        self.cache = (hs.shape, rhs.shape)
        scores = py.hstack((exp_scores, org_scores, cont_scores))
        return scores

    def forward(self, x, t):
        scores = self.predict(x)
        loss = self.loss_layer.forward(scores, t)
        return loss

    def backward(self, dout=1):
        hs_shape, rhs_shape = self.cache

        dscore = self.loss_layer.backward(dout)

        dexp_scores = dscore[:9][py.newaxis].repeat(rhs_shape[0], axis=0) / rhs_shape[0]
        dorg_scores = dscore[9:21][py.newaxis].repeat(rhs_shape[0], axis=0) / rhs_shape[0]
        dcont_scores = dscore[21:33][py.newaxis].repeat(rhs_shape[0], axis=0) / rhs_shape[0]

        drhs = py.zeros(rhs_shape, dtype='f')
        drhs += self.exp_affine_layer.backward(dexp_scores)[:, :rhs_shape[-1]]
        drhs += self.org_affine_layer.backward(dorg_scores)[:, :rhs_shape[-1]]
        drhs += self.cont_affine_layer.backward(dcont_scores)[:, :rhs_shape[-1]]

        dhs = drhs.reshape(hs_shape)
        dembed_xs = self.time_lstm_layer.backward(dhs)
        self.time_embedding_layer.backward(dembed_xs)


    def reset_state(self):
        self.time_lstm_layer.reset_state()

    # todo: 데이터 스케일 전처리
    @classmethod
    def get_x_t_list_from_processed_data(cls, data_list, time_size, load_pickle=True, save_pickle=False, pickle_name='eval_model_args.p'):
        if load_pickle:
            args_list = load_data(pickle_name)
            if args_list is not None:
                return args_list

        args_list = []

        for data in data_list:
            xs = py.array(sum(data['paragraph'], []))
            split_amount = math.ceil(len(xs) / time_size)
            xs = py.resize(xs, (1, split_amount * time_size)).reshape(-1, time_size)  # time size에 맞는 크기가 될 때까지 내용을 반복시킨다.

            exp_weight, org_weight, cont_weight = data['weight']['exp'], data['weight']['org'], data['weight']['cont']
            # 대분류 가중치를 하위의 소분류 가중치에 곱한다.
            score_metrics = [exp_weight['exp'] * py.array([exp_weight['exp_grammar'], exp_weight['exp_vocab'], exp_weight['exp_style'], data['corr_count'] * exp_weight['exp_grammar']]),  # 문법 점수 계산에 교정 횟수를 포함
                             org_weight['org'] * py.array([org_weight['org_paragraph'], org_weight['org_essay'], org_weight['org_coherence'], org_weight['org_quantity']]),
                             cont_weight['con'] * py.array([cont_weight['con_clearance'], cont_weight['con_novelty'], cont_weight['con_prompt'], cont_weight['con_description']])]

            t = sum(data['score']['exp'], []) + sum(data['score']['org'], []) + sum(data['score']['cont'], [])
            t = py.array(t)

            x = (xs, score_metrics)
            args_list.append([x, t])

        if save_pickle:
            save_data(pickle_name, args_list)

        return args_list

