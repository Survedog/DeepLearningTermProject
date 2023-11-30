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

    return corpus, id_to_word, word_to_id


def create_context_and_target(corpus):
    assert(len(corpus) >= 3)
    context, target = [], []

    for target_idx in range(1, len(corpus)-1):
        context.append((corpus[target_idx - 1], corpus[target_idx + 1]))
        target.append(corpus[target_idx])

    return context, target


def get_one_hot_encoding(num, array_size):
    encoded = np.zeros(array_size)
    encoded[num] = 1
    return encoded


def get_class_cross_entropy(y, t):
    '''
    분류 문제에서의 Cross Entropy
    t: 정답 레이블 인덱스
    '''
    if y.ndim == 1:
        return -np.log(y[t])
    else:
        return -np.log(y[range(t.size), t])
