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
        text = "안녕하세요. 저는 프로그래머입니다. 만나서 반갑습니다."
        corpus, id_to_word, word_to_id = create_corpus_and_dict(text)

        parsed_text = text_parser.morphs(text)

        for idx, word_id in enumerate(corpus):
            word = parsed_text[idx]
            self.assertEqual(word, id_to_word[word_id])
            self.assertEqual(word_id, word_to_id[word])

    def test_create_context_and_target(self):
        corpus = [1, 2, 1, 3, 2, 4]
        context, target = create_context_and_target(corpus)
        self.assertListEqual(context, [(1, 1), (2, 3), (1, 2), (3, 4)])
        self.assertListEqual(target, [2, 1, 3, 2])

    def test_get_one_hot_encoding(self):
        encoded = get_one_hot_encoding(2, 5)
        answer = np.array([0, 0, 1, 0, 0])
        self.assertTrue(np.array_equal(encoded, answer))

        encoded = get_one_hot_encoding(0, 100)
        answer = np.zeros(100)
        answer[0] = 1
        self.assertTrue(np.array_equal(encoded, answer))

        encoded = get_one_hot_encoding(99, 100)
        answer = np.zeros(100)
        answer[99] = 1
        self.assertTrue(np.array_equal(encoded, answer))

