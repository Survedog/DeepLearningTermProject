import unittest
from utils import *

if Config.USE_GPU:
    import cupy as np
else:
    import numpy as np


class UtilsTests(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(-1), 1 / (1 + np.e))
        self.assertTrue(np.array_equal(sigmoid(np.array([1, 2])),
                                       np.array([1 / (1 + np.exp(-1)),
                                                 1 / (1 + np.exp(-2))])))

    def test_create_corpus_and_dict(self):
        corpus, id_to_word, word_to_id = create_corpus_and_dict("안녕하세요. 저는 프로그래머입니다. 만나서 반갑습니다.")
        print(corpus)
        print(id_to_word)
        print(word_to_id)