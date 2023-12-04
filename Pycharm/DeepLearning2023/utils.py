from config import Config
from konlpy.tag import Kkma

if Config.USE_GPU:
    import cupy as np
else:
    import numpy as np

text_parser = Kkma()


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def create_corpus_and_dict(text):
    """
    :param text: corpus를 만들 문장
    :return: corpus(list), id_to_word(dict), word_to_id(dict)
    """
    parsed_text = text_parser.morphs(text)
    corpus = []
    id_to_word = {}
    word_to_id = {}

    id_count = 0
    for word in parsed_text:
        if word not in word_to_id:
            id_to_word[id_count] = word
            word_to_id[word] = id_count
            id_count += 1

        corpus.append(word_to_id[word])

    return np.array(corpus), id_to_word, word_to_id


def create_context_and_target(corpus):
    """
    :param corpus: 문맥과 타겟 데이터를 만들기 위한 단어 id 배열
    :return: context(list), target(list)
    """
    assert(len(corpus) >= 3)
    context, target = [], []

    for target_idx in range(1, len(corpus)-1):
        context.append((corpus[target_idx - 1], corpus[target_idx + 1]))
        target.append(corpus[target_idx])

    return np.array(context), np.array(target)


def get_one_hot_encoding(num, array_size):
    encoded = np.zeros(array_size)
    encoded[num] = 1
    return encoded


def get_class_cross_entropy(y, t):
    """
    분류 문제에서의 Cross Entropy
    :param y: 각 클래스의 확률 (N x S x 클래스 수)
    :param t: 정답 클래스 레이블 (N x S)
    :return: Cross Entropy 손실 값 (N x S)
    """

    batch_index = np.repeat(np.arange(0, t.shape[0]), t.shape[1]).reshape(t.shape)
    sample_index = np.resize(np.arange(0, t.shape[1]), t.shape)
    return -np.log(y[batch_index, sample_index, t])
