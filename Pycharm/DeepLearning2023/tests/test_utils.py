import unittest

import numpy as np

from utils import *


class UtilsTests(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(-1), 1 / (1 + np.e))

        # 1d array
        answer = np.array([1 / (1 + np.exp(-1)),
                           1 / (1 + np.exp(-2))])
        self.assertTrue(
            np.array_equal(sigmoid(np.array([1, 2])), answer)
        )

        # 2d array
        answer = np.array([[1 / (1 + np.exp(-1)), 1 / (1 + np.exp(-2))],
                           [1 / (1 + np.exp(-3)), 1 / (1 + np.exp(-4))]])
        self.assertTrue(
            np.array_equal(sigmoid(np.array([[1, 2], [3, 4]])), answer)
        )

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

        correct_context = np.array([(1, 1), (2, 3), (1, 2), (3, 4)])
        self.assertTrue(np.array_equal(context, correct_context))

        correct_target = np.array([2, 1, 3, 2])
        self.assertTrue(np.array_equal(target, correct_target))

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

    def test_get_class_cross_entropy(self):
        # 3차원 배열 (배치 처리)
        # N = 2, S = 2일 때
        y = np.array([[[0.8, 0.2],
                       [0.3, 0.7]],

                      [[0.5, 0.5],
                       [0.6, 0.4]]])
        t = np.array([[0, 1],
                      [0, 0]])

        self.assertTrue(
            np.array_equal(get_class_cross_entropy(y, t),
                           -np.log([[0.8, 0.7],
                                    [0.5, 0.6]])))

        # N = 1, S = 2일 때
        y = np.array([[[0.7, 0.3]]])
        t = np.array([[0]])

        self.assertTrue(
            np.array_equal(get_class_cross_entropy(y, t),
                           ([[-np.log(0.7)]])))

        # N = 2, S = 3일 때
        y = np.array([[[0.8, 0.2],
                       [0.5, 0.5],
                       [0.9, 0.1]],

                      [[0.5, 0.5],
                       [0.2, 0.8],
                       [0.6, 0.4]]])
        t = np.array([[0, 0, 1],
                      [0, 1, 0]])

        self.assertTrue(
            np.array_equal(get_class_cross_entropy(y, t),
                           -np.log([[0.8, 0.5, 0.1],
                                    [0.5, 0.8, 0.6]])))

        # N = 3, S = 2일 때
        y = np.array([[[0.8, 0.2],
                       [0.5, 0.5]],

                      [[0.5, 0.5],
                       [0.2, 0.8]],

                      [[0.4, 0.6],
                       [0.7, 0.3]]])
        t = np.array([[0, 1],
                      [1, 0],
                      [0, 0]])

        self.assertTrue(
            np.array_equal(get_class_cross_entropy(y, t),
                           -np.log([[0.8, 0.5],
                                    [0.5, 0.2],
                                    [0.4, 0.7]])))
