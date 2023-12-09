import unittest
from word2vec.cbow_model import *
from common.utils import create_corpus_and_dict, create_context_and_target


class CBowModelTests(unittest.TestCase):

    def test_forward_and_backward(self):
        corpus, id_to_word, _ = create_corpus_and_dict('오늘은 이미 밥을 먹었습니다. 바로 초밥을 먹었습니다.')
        contexts, true_label = create_context_and_target(corpus)

        hidden_size = 3
        vocab_size = len(id_to_word)
        weight_in = py.random.rand(vocab_size, hidden_size)
        weight_out = py.random.rand(vocab_size, hidden_size)

        cbow = CBowModel(corpus=corpus,
                         vocab_size=vocab_size,
                         hidden_size=hidden_size,
                         sample_size=5,
                         weight_in=weight_in,
                         weight_out=weight_out)

        cbow.forward(contexts[0:1], true_label[0:1])
        dout = py.ones(1)
        cbow.backward(dout)
        dWin, dWout = cbow.grads[0], cbow.grads[1]

        # dWin에서 입력 문맥의 기울기만 갱신되는지 확인
        dWin_sum = py.sum(dWin, axis=1)
        input_mask = py.zeros_like(dWin_sum, dtype=bool)
        input_mask[contexts[0:1]] = True
        self.assertTrue(py.all(dWin_sum[input_mask]))
        self.assertFalse(py.any(dWin_sum[~input_mask]))

        # 배치 입력의 경우
        cbow.forward(contexts[0:3], true_label[0:3])
        dout = py.ones(3)
        cbow.backward(dout)
        dWin, dWout = cbow.grads[0], cbow.grads[1]

        # 형상 일치 확인
        self.assertEqual(dWin.shape, weight_in.shape)
        self.assertEqual(dWout.shape, weight_out.shape)

        # dWin에서 입력한 배치 내 문맥들의 기울기만 갱신되는지 확인
        dWin_sum = py.sum(dWin, axis=1)
        input_mask = py.zeros_like(dWin_sum, dtype=bool)
        input_mask[contexts[0:3]] = True
        self.assertTrue(py.all(dWin_sum[input_mask]))
        self.assertFalse(py.any(dWin_sum[~input_mask]))
