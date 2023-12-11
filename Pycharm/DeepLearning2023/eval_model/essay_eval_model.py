from common.base_layer import LayerBase
from common.affine_layer import AffineLayer
from common.utils import py, load_data, save_data
from common.loss_layers import SSELossLayer
import math


class EssayEvalModel(LayerBase):

    def __init__(self, rnn_model, time_size=50, fit_premade_models=False):
        super().__init__()
        self.cache = None

        affine_input_size = time_size + 4
        exp_metrics_amount, org_metrics_amount, cont_metrics_amount = 3, 4, 4

        randn = py.random.randn
        exp_affine_weight = randn(affine_input_size, exp_metrics_amount * 3, dtype='f')
        exp_affine_bias = randn(exp_metrics_amount * 3, dtype='f')
        org_affine_weight = randn(affine_input_size, org_metrics_amount * 3, dtype='f')
        org_affine_bias = randn(org_metrics_amount * 3, dtype='f')
        cont_affine_weight = randn(affine_input_size, cont_metrics_amount * 3, dtype='f')
        cont_affine_bias = randn(cont_metrics_amount * 3, dtype='f')

        self.rnn_model = rnn_model
        self.exp_affine_layer = AffineLayer(exp_affine_weight, exp_affine_bias)
        self.org_affine_layer = AffineLayer(org_affine_weight, org_affine_bias)
        self.cont_affine_layer = AffineLayer(cont_affine_weight, cont_affine_bias)

        self.layers = [self.rnn_model, self.org_affine_layer, self.cont_affine_layer, self.exp_affine_layer]
        self.loss_layer = SSELossLayer()

        train_from = 0
        if not fit_premade_models:
            train_from = 1

        for i in range(train_from, len(self.layers)):
            layer = self.layers[i]
            self.params += layer.params
            self.grads += layer.grads


    def predict(self, xs, score_metrics):
        hs = self.rnn_model.predict(xs)
        fhs = hs.flatten()

        org_x = py.hstack((fhs, score_metrics[0]))
        cont_x = py.hstack((fhs, score_metrics[1]))
        exp_x = py.hstack((fhs, score_metrics[2]))

        org_scores = self.org_affine_layer.forward(org_x)
        cont_scores = self.cont_affine_layer.forward(cont_x)
        exp_scores = self.exp_affine_layer.forward(exp_x)

        self.cache = (hs.shape, fhs.shape[-1])
        return org_scores, cont_scores, exp_scores

    def forward(self, xs, score_weights, t):
        scores = self.predict(xs, score_weights)
        loss = self.loss_layer.forward(scores, t)
        return loss

    def backward(self, dout=1):
        hs_shape, fhs_len = self.cache

        dscore = self.loss_layer.backward(dout)

        dhs = py.zeros(fhs_len, dtype='f')
        dhs += self.exp_affine_layer.backward(dscore)[:fhs_len]
        dhs += self.cont_affine_layer.backward(dscore)[:fhs_len]
        dhs += self.org_affine_layer.backward(dscore)[:fhs_len]

        dhs = dhs.reshape(hs_shape)
        self.rnn_model.backward(dhs)

    def reset_state(self):
        self.rnn_model.reset_state()

    @classmethod
    def get_args_from_processed_data(cls, data, time_size=50, load_pickle=True, save_pickle=False, pickle_name='eval_model_args.p'):
        if load_pickle:
            args = load_data(pickle_name)
            if args is not None:
                return args

        xs = []
        for paragraph in data['paragraph']:
            split_amount = math.ceil(len(paragraph) / time_size)

            paragraph = py.array(paragraph)
            paragraph.resize((1, split_amount * time_size))  # time size에 맞는 크기가 될 때까지 내용을 반복시킨다.
            paragraph = paragraph.reshape(-1, time_size)
            xs.append(paragraph)
        xs = py.array(xs)

        org_weight, cont_weight, exp_weight = data['weight']['org'], data['weight']['cont'], data['weight']['exp']
        # 대분류 가중치를 그에 해당하는 소분류 가중치에 곱한다.
        score_metrics = [exp_weight['exp'] * py.array([exp_weight['exp_grammar'], exp_weight['exp_vocab'], exp_weight['exp_style'], data['corr_count']]),  # 문법 점수 계산에 교정 횟수를 포함
                         org_weight['org'] * py.array([org_weight['org_paragraph'], org_weight['org_essay'], org_weight['org_coherence'], org_weight['org_quantity']]),
                         cont_weight['con'] * py.array([cont_weight['con_clearance'], cont_weight['con_novelty'], cont_weight['con_prompt'], cont_weight['con_description']])]

        t = []
        for score_list in data['score']:
            t += sum(score_list, [])

        if save_pickle:
            args = xs, score_metrics, t
            save_data(pickle_name, args)

        return xs, score_metrics, t

