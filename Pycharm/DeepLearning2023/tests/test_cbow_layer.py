import unittest
from cbow_layer import *
from utils import create_corpus_and_dict, create_context_and_target


class CBowLayerTests(unittest.TestCase):

    def test_forward_and_backward(self):
        corpus, id_to_word, _ = create_corpus_and_dict('오늘은 이미 밥을 먹었습니다.')
        contexts, true_label = create_context_and_target(corpus)

        hidden_size = 3
        vocab_size = len(id_to_word)
        weight_in = np.random.rand(vocab_size, hidden_size)
        weight_out = np.random.rand(vocab_size, hidden_size)

        cbow = CBowLayer(corpus=corpus,
                         window_size=1,
                         vocab_size=4,
                         hidden_size=hidden_size,
                         sample_size=5,
                         weight_in=weight_in,
                         weight_out=weight_out)

        cbow.forward(contexts[0:1], true_label[0:1])
        dout = np.ones(1)
        dWin, dWout = cbow.backward(dout)

        # dWin에서 입력 문맥의 기울기만 갱신되는지 확인
        dWin_sum = np.sum(dWin, axis=1)
        mask = np.zeros_like(dWin_sum)
        mask[contexts[0:1]] = True
        self.assertTrue(np.all(dWin_sum[mask]))

        mask = not mask
        self.assertFalse(np.any(dWin_sum[mask]))

        # dWout에서 입력 true_label의 기울기만 갱신되는지 확인
        dWout_sum = np.sum(dWout, axis=1)
        mask = np.zeros_like(dWout_sum)
        mask[true_label[0]] = True
        self.assertTrue(dWout_sum[mask])

        mask = not mask
        self.assertFalse(np.any(dWout_sum[mask]))

        # 배치 입력의 경우
        cbow.forward(contexts[0:3], true_label[0:3])
        dout = np.ones(3)
        dWin, dWout = cbow.backward(dout)

        # 형상 일치 확인
        self.assertEqual(dWin.shape, weight_in.shape)
        self.assertEqual(dWout.shape, weight_out.shape)

        # dWin에서 입력한 배치 내 문맥들의 기울기만 갱신되는지 확인
        dWin_sum = np.sum(dWin, axis=1)
        mask = np.zeros_like(dWin_sum)
        mask[contexts[0:3]] = True
        self.assertTrue(np.all(dWin_sum[mask]))

        mask = not mask
        self.assertFalse(np.any(dWin_sum[mask]))

        # dWout에서 입력한 배치 내 true_label들의 기울기만 갱신되는지 확인
        dWout_sum = np.sum(dWout, axis=1)
        mask = np.zeros_like(dWout_sum)
        mask[true_label[0:3]] = True
        self.assertTrue(dWout_sum[mask])

        mask = not mask
        self.assertFalse(np.any(dWout_sum[mask]))
