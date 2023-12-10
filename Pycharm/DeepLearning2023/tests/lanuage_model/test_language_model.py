import unittest
from common.utils import create_corpus_and_dict
from lang_model.language_model import *


class LanguageModelTests(unittest.TestCase):

    def test_forward_and_backward(self):

        corpus, id_to_word, word_to_id = create_corpus_and_dict('안녕하세요. 만나서 반갑습니다. 저도 반갑습니다. 다음에 또 봅시다.')

        vocab_size = len(id_to_word)
        wordvec_size = 9
        hidden_size = 6

        model = LanguageModel(vocab_size=vocab_size,
                              wordvec_size=wordvec_size,
                              hidden_size=hidden_size)

        # 1차원 입력
        xs = corpus[:-1]
        ts = corpus[1:]

        loss = model.forward(xs, ts)
        model.backward()

        # 2차원 입력 (배치 처리)
        batch_size = 2
        time_size = (len(corpus) - 1) // 2

        xs = py.empty((batch_size, time_size), dtype=int)
        ts = py.empty((batch_size, time_size), dtype=int)

        for i in range(batch_size):
            xs[i] = corpus[i * time_size: (i+1) * time_size]
            ts[i] = corpus[i * time_size + 1: (i + 1) * time_size + 1]

        loss = model.forward(xs, ts)
        model.backward()
