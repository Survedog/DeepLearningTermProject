import unittest
from common.utils import *


class UtilsTests(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(-1), 1 / (1 + py.e))

        # 1d array
        answer = py.array([1 / (1 + py.exp(-1)),
                           1 / (1 + py.exp(-2))])
        self.assertTrue(
            py.array_equal(sigmoid(py.array([1, 2])), answer)
        )

        # 2d array
        answer = py.array([[1 / (1 + py.exp(-1)), 1 / (1 + py.exp(-2))],
                           [1 / (1 + py.exp(-3)), 1 / (1 + py.exp(-4))]])
        self.assertTrue(
            py.array_equal(sigmoid(py.array([[1, 2], [3, 4]])), answer)
        )

    def test_softmax(self):

        # 1차원 입력
        values = py.array([0, 1, 2, 3, 4])
        result = softmax(values)

        exp_sum = 1 + py.e + py.e ** 2 + py.e ** 3 + py.e ** 4
        correct_result = py.array([1, py.e, py.e ** 2, py.e ** 3, py.e ** 4]) / exp_sum
        self.assertTrue(py.allclose(result, correct_result))

        # 2차원 입력
        values = py.array([[0, 1, 2],
                           [3, 4, 5]])
        result = softmax(values)

        exp_sum = [1 + py.e + py.e ** 2,
                   py.e ** 3 + py.e ** 4 + py.e ** 5]
        correct_result = py.array([[1, py.e, py.e ** 2],
                                   [py.e ** 3, py.e ** 4, py.e ** 5]])
        correct_result[0] /= exp_sum[0]
        correct_result[1] /= exp_sum[1]
        self.assertTrue(py.allclose(result, correct_result))

        # 3차원 입력
        values = py.array([[[0, 1, 2],
                            [3, 4, 5]],

                           [[6, 7, 8],
                            [9, 10, 11]]])
        result = softmax(values)

        exp_sum = [[1 + py.e + py.e ** 2,
                    py.e ** 3 + py.e ** 4 + py.e ** 5],
                   [py.e ** 6 + py.e ** 7 + py.e ** 8,
                    py.e ** 9 + py.e ** 10 + py.e ** 11]]
        correct_result = py.array([[[1, py.e, py.e ** 2],
                                    [py.e ** 3, py.e ** 4, py.e ** 5]],
                                   [[py.e ** 6, py.e ** 7, py.e ** 8],
                                    [py.e ** 9, py.e ** 10, py.e ** 11]]])
        correct_result[0, 0] /= exp_sum[0][0]
        correct_result[0, 1] /= exp_sum[0][1]
        correct_result[1, 0] /= exp_sum[1][0]
        correct_result[1, 1] /= exp_sum[1][1]
        self.assertTrue(py.allclose(result, correct_result))

    def test_create_corpus_and_dict(self):
        text = "안녕하세요. 저는 프로그래머입니다. 만나서 반갑습니다."
        corpus, id_to_word, word_to_id = create_corpus_and_dict(text)

        if Config.USE_GPU:
            corpus = py.asnumpy(corpus)
        parsed_text = text_parser.morphs(text)

        for idx, word_id in enumerate(corpus):
            word = parsed_text[idx]
            self.assertEqual(word, id_to_word[word_id])
            self.assertEqual(word_id, word_to_id[word])

        text = ["안녕하세요. 저는 프로그래머입니다.", "만나서 반갑습니다."]
        corpus2, id_to_word2, word_to_id2 = create_corpus_and_dict(text)
        if Config.USE_GPU:
            corpus2 = py.asnumpy(corpus2)

        self.assertTrue(py.array_equal(corpus, corpus2))
        self.assertDictEqual(id_to_word, id_to_word2)
        self.assertDictEqual(word_to_id, word_to_id2)

    def test_create_context_and_target(self):
        corpus = py.array([1, 2, 1, 3, 2, 4])
        context, target = create_context_and_target(corpus)

        correct_context = py.array([(1, 1), (2, 3), (1, 2), (3, 4)])
        self.assertTrue(py.array_equal(context, correct_context))

        correct_target = py.array([2, 1, 3, 2])
        self.assertTrue(py.array_equal(target, correct_target))

    def test_get_one_hot_encoding(self):
        # 정수 입력
        encoded = get_one_hot_encoding(2, 5)
        answer = py.array([0, 0, 1, 0, 0])
        self.assertTrue(py.array_equal(encoded, answer))

        encoded = get_one_hot_encoding(0, 100)
        answer = py.zeros(100)
        answer[0] = 1
        self.assertTrue(py.array_equal(encoded, answer))

        encoded = get_one_hot_encoding(99, 100)
        answer = py.zeros(100, dtype=int)
        answer[99] = 1
        self.assertTrue(py.array_equal(encoded, answer))

        # 1차원 입력
        nums = py.array([0, 8, 2])
        encoded = get_one_hot_encoding(nums, 10)
        answer = py.zeros((3, 10), dtype=int)
        answer[0, 0], answer[1, 8], answer[2, 2] = 1, 1, 1
        self.assertTrue(py.array_equal(encoded, answer))

        # 2차원 입력
        nums = py.array([[0, 8, 2],
                         [1, 4, 3]])
        encoded = get_one_hot_encoding(nums, 10)
        answer = py.zeros((2, 3, 10), dtype=int)
        answer[0, 0, 0], answer[0, 1, 8], answer[0, 2, 2] = 1, 1, 1
        answer[1, 0, 1], answer[1, 1, 4], answer[1, 2, 3] = 1, 1, 1
        self.assertTrue(py.array_equal(encoded, answer))

    def test_get_class_cross_entropy(self):
        # 입력: 2 x 2 x 2
        y = py.array([[[0.8, 0.2],
                       [0.3, 0.7]],

                      [[0.5, 0.5],
                       [0.6, 0.4]]])
        t = py.array([[0, 1],
                      [0, 0]])

        self.assertTrue(
            py.array_equal(get_class_cross_entropy(y, t),
                           -py.log(py.array([[0.8, 0.7],
                                             [0.5, 0.6]]))))

        # 입력: 1 x 3
        y = py.array([0.7, 0.2, 0.1])
        t = py.array([0])

        self.assertTrue(
            py.array_equal(get_class_cross_entropy(y, t), py.array([-py.log(0.7)])))

        #  입력: 2 x 3 x 1
        y = py.array([[[0.8],
                       [0.5],
                       [0.9]],

                      [[0.5],
                       [0.2],
                       [0.6]]])
        t = py.array([[0, 0, 0],
                      [0, 0, 0]])

        self.assertTrue(
            py.array_equal(get_class_cross_entropy(y, t),
                           -py.log(py.array([[0.8, 0.5, 0.9],
                                             [0.5, 0.2, 0.6]]))))

        #  입력: 3 x 2 x 2
        y = py.array([[[0.8, 0.2],
                       [0.5, 0.5]],

                      [[0.5, 0.5],
                       [0.2, 0.8]],

                      [[0.4, 0.6],
                       [0.7, 0.3]]])
        t = py.array([[0, 1],
                      [1, 0],
                      [0, 0]])

        self.assertTrue(
            py.array_equal(get_class_cross_entropy(y, t),
                           -py.log(py.array([[0.8, 0.5],
                                             [0.5, 0.2],
                                             [0.4, 0.7]]))))

    def test_get_sum_of_squares_error(self):
        # 1차원 입력
        x = py.array([5.0, 4.5, 3.2, 2.1, 1.9])
        t = py.array([4.5, 4.5, 2.2, 1.1, -0.1])

        result = get_sum_of_squares_error(x, t)
        correct_result = (0.25 + 1 + 1 + 4) / 2
        self.assertAlmostEqual(result, correct_result)

        # 2차원 입력
        x = py.array([[5.0, 4.5, 3.2],
                      [1.0, -2.5, -0.5]])
        t = py.array([[5.0, 4.5, 3.2],
                      [0.0, 2.5, -0.5]])

        result = get_sum_of_squares_error(x, t)
        correct_result = py.array([0.0, 26]) / 2
        self.assertTrue(py.allclose(result, correct_result))


    def test_load_essay_data_list(self):
        train_data = load_essay_data_list(load_test_data=False, load_pickle=False)
        test_data = load_essay_data_list(load_test_data=True, load_pickle=False)
        self.assertTrue(train_data)
        self.assertTrue(test_data)

        train_data = load_essay_data_list(load_test_data=False, load_pickle=True)
        test_data = load_essay_data_list(load_test_data=True, load_pickle=True)
        self.assertTrue(train_data)
        self.assertTrue(test_data)

    def test_create_essay_corpus_and_dict(self):
        corpus, id_to_word, word_to_id = create_essay_corpus_and_dict(load_pickle=False, save_pickle=False, batch_size=100)

        diff = py.setdiff1d(corpus, py.array(list(id_to_word.keys())))
        self.assertTrue(len(diff) == 0)

        for word_id, word in id_to_word.items():
            self.assertEqual(word_id, word_to_id[word])

    def test_paragraph_text_to_ids(self):
        text_list = ['안녕하세요, 만나서 반갑습니다.', '그러면 연락 기다리고 있겠습니다.']
        corpus, id_to_word, word_to_id = create_corpus_and_dict(text_list)
        result = paragraph_text_to_ids(text_list, word_to_id)
        self.assertTrue(py.array_equal(corpus, result))

    def test_get_index_value_tuple(self):
        # 1차원 입력
        arr = py.array([5, 4, 3, 2])
        answer = py.array([[0, 1, 2, 3]])
        answer = tuple(answer)

        for a, b in zip(get_ndarray_index(arr), answer):
            self.assertTrue(py.array_equal(a, b))

        # 2차원 입력
        arr = py.array([[5, 4, 3],
                        [3, 4, 5]])
        answer = py.array([[0, 0, 0, 1, 1, 1],
                           [0, 1, 2, 0, 1, 2]])
        answer = tuple(answer)

        for a, b in zip(get_ndarray_index(arr), answer):
            self.assertTrue(py.array_equal(a, b))

        # 3차원 입력
        arr = py.array([[[5, 4, 3],
                         [3, 4, 5]],

                        [[4, 7, 3],
                         [1, 5, 3]]])
        answer = py.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                           [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]])
        answer = tuple(answer)

        for a, b in zip(get_ndarray_index(arr), answer):
            self.assertTrue(py.array_equal(a, b))


