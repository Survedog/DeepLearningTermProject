import unittest
from word2vec.skipgram_model import *
from common.utils import create_corpus_and_dict, create_context_and_target


class SkipgramModelTests(unittest.TestCase):

    def test_forward_and_backward(self):
        corpus, id_to_word, _ = create_corpus_and_dict('오늘은 이미 밥을 먹었습니다. 바로 초밥을 먹었습니다.')
        targets, context = create_context_and_target(corpus)  # context와 target을 서로 뒤바꾼다.

        wordvec_size = 3
        vocab_size = len(id_to_word)

        weight_in = py.random.randn(vocab_size, wordvec_size)
        weight_out_list = [py.random.randn(vocab_size, wordvec_size) for _ in range(targets.shape[-1])]

        skipgram = SkipgramModel(corpus=corpus,
                                 vocab_size=vocab_size,
                                 wordvec_size=wordvec_size,
                                 sample_size=5,
                                 weight_in=weight_in,
                                 weight_out_list=weight_out_list)

        skipgram.forward(context[0:1], targets[0:1])
        dout = py.ones(1)
        skipgram.backward(dout)
        dWin, *dWout_list = skipgram.grads

        # 형상 일치 확인
        self.assertEqual(dWin.shape, weight_in.shape)
        for i in range(targets.shape[-1]):
            self.assertEqual(dWout_list[i].shape, weight_out_list[i].shape)

        # dWin에서 입력 문맥의 기울기만 갱신되는지 확인
        dWin_sum = py.sum(dWin, axis=1)
        input_mask = py.zeros_like(dWin_sum, dtype=bool)
        input_mask[context[0:1]] = True
        self.assertTrue(py.all(dWin_sum[input_mask]))
        self.assertFalse(py.any(dWin_sum[~input_mask]))

        # 배치 입력의 경우
        skipgram.forward(context[0:3], targets[0:3])
        dout = py.ones(3)
        skipgram.backward(dout)
        dWin, *dWout_list = skipgram.grads

        # 형상 일치 확인
        self.assertEqual(dWin.shape, weight_in.shape)
        for i in range(targets.shape[-1]):
            self.assertEqual(dWout_list[i].shape, weight_out_list[i].shape)

        # dWin에서 입력한 배치 내 문맥들의 기울기만 갱신되는지 확인
        dWin_sum = py.sum(dWin, axis=1)
        input_mask = py.zeros_like(dWin_sum, dtype=bool)
        input_mask[context[0:3]] = True
        self.assertTrue(py.all(dWin_sum[input_mask]))
        self.assertFalse(py.any(dWin_sum[~input_mask]))
