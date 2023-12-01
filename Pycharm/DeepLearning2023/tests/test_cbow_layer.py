import unittest
from cbow_layer import *
from utils import create_corpus_and_dict, create_context_and_target


class CBowLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        corpus, id_to_word, _ = create_corpus_and_dict('오늘은 이미 밥을 먹었습니다.')
        contexts, true_label = create_context_and_target(corpus)

        hidden_size = 3
        weight_in = np.random.rand(corpus.size(), hidden_size)
        weight_out = np.random.rand(corpus.size(), hidden_size)

        cbow = CBowLayer(corpus=corpus,
                         window_size=1,
                         vocab_size=5,
                         hidden_size=hidden_size,
                         sample_size=5,
                         weight_in=weight_in,
                         weight_out=weight_out)

        cbow.forward(contexts[0:1], true_label[0:1])
        dout = np.ones_like(true_label[0:1])
        dWin, dWout = cbow.backward(dout)

        # dWeight_in에서 현재 문맥의 기울기만 갱신되는지 확인
        dWin_Sum = np.sum(dWin, axis=1)
        mask = np.zeros_like(dWin_Sum)
        mask[contexts[0]] = True
        self.assertTrue(np.all(dWin_Sum[mask]))

        mask = not mask
        self.assertFalse(np.any(dWin_Sum[mask]))

        # dWeight_out에서 현재 true_label의 기울기만 갱신되는지 확인
        dWout_Sum = np.sum(dWout, axis=1)
        mask = np.zeros_like(dWout_Sum)
        mask[true_label[0]] = True
        self.assertTrue(dWout_Sum[mask])

        mask = not mask
        self.assertFalse(np.any(dWout_Sum[mask]))

        # 배치 처리의 경우
        cbow.forward(contexts[0:2], true_label[0:2])
        dout = np.ones_like(true_label[0:2])

        dWin, dWout = cbow.backward(dout)
        self.assertEqual(dWin.shape, weight_in.shape)
        self.assertEqual(dWout.shape, weight_out.shape)
