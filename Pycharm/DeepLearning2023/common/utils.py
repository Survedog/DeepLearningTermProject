from config import Config
from konlpy.tag import Komoran
from pathlib import Path
import json
import pickle
import numpy

if Config.USE_GPU:
    import cupy as py
else:
    import numpy as py

text_parser = Komoran()


def sigmoid(values):
    return 1 / (1 + py.exp(-values))


def softmax(values):
    values = py.exp(values)
    sum = py.sum(values, axis=-1)
    sum = py.expand_dims(sum, axis=-1)

    return values / sum


def create_corpus_and_dict(text_list):
    """
    :param text_list: corpus를 만들 문장 리스트
    :return: corpus(list), id_to_word(dict), word_to_id(dict)
    """
    if type(text_list) != list:
        text_list = [text_list]

    corpus = []
    id_to_word = {}
    word_to_id = {}
    id_count = 0

    for i, text in enumerate(text_list):
        print('Create corpus from text[%d/%d] - id_count: %d' % (i, len(text_list), id_count))
        try:
            parsed_text = text_parser.morphs(text)
        except UnicodeDecodeError:
            print('Error while parsing text[%d/%d]' % (i, len(text_list)))
            continue

        for word in parsed_text:
            if word not in word_to_id:
                id_to_word[id_count] = word
                word_to_id[word] = id_count
                id_count += 1

            corpus.append(word_to_id[word])

    return py.array(corpus), id_to_word, word_to_id


def create_context_and_target(corpus):
    """
    :param corpus: 문맥과 타겟 데이터를 만들기 위한 단어 id 배열
    :return: context(list), target(list)
    """
    assert(len(corpus) >= 3)
    context = py.stack((corpus[0:-2], corpus[2:]), axis=-1)
    target = py.array(corpus[1:-1])

    return context, target


def get_one_hot_encoding(num, array_size):
    if isinstance(num, int):
        encoded = py.zeros(array_size)
        encoded[num] = 1
    else:
        encoded = py.zeros((num.shape[0], array_size))
        encoded[range(num.shape[0]), num] = 1
    return encoded


def get_class_cross_entropy(y, t):
    """
    분류 문제에서의 Cross Entropy
    N: 배치 개수, S: 샘플 개수
    :param y: 각 클래스의 확률 (N x S x 클래스 수)
    :param t: 정답 클래스 레이블 (N x S)
    :return: Cross Entropy 손실 값 (N x S)
    """
    original_t_shape = t.shape
    if t.ndim == 1:
        t = t.reshape((1, -1))
    if y.ndim < 3:
        y = y.reshape((1,) * (3 - y.ndim) + y.shape)

    batch_index = py.repeat(py.arange(t.shape[0]), t.shape[1]).reshape(t.shape)
    sample_index = py.resize(py.arange(t.shape[1]), t.shape)
    return (-py.log(y[batch_index, sample_index, t])).reshape(original_t_shape)


def save_data(pickle_name, data):
    pickle_path = Config.PICKLE_PATH.joinpath(pickle_name)
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)


def load_data(pickle_name):
    path = Config.PICKLE_PATH.joinpath(pickle_name)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print('Error - load_pickle: pickle file does not exist.')
        return None


def load_essay_data_list(load_test_data, load_pickle=True):
    pickle_name = 'essay_data_' + ('test.p' if load_test_data else 'train.p')
    if load_pickle:
        data_list = load_data(pickle_name)
        if data_list is not None:
            return data_list

    data_list = []
    json_files = Config.DATA_DIR_PATH.joinpath('test_data' if load_test_data else 'train_data').glob('**/*.json')

    for file_path in json_files:
        with open(file_path, "r", encoding='UTF-8') as f:
            data_list.append(json.load(f))

    save_data(pickle_name, data_list)
    return data_list


def create_essay_corpus_and_dict(load_pickle=True, batch_size=100):
    pickle_name = 'corpus_and_dicts.p'
    if load_pickle:
        data = load_data(pickle_name)
        if data is not None:
            return data

    essay_data_list = load_essay_data_list(load_test_data=False, load_pickle=True)
    text_list = []
    text_batch = []

    for essay_data in essay_data_list:
        for paragraph in essay_data['paragraph']:
            text = paragraph['paragraph_txt'].replace('\n', '').replace('\r', '').replace('\t', '')
            text_batch += text.split('#@문장구분#')[0:-1]

            if len(text_batch) % batch_size == 0 and len(text_batch) != 0:
                text_list.append(''.join(text_batch))
                text_batch = []
    text_list.append(''.join(text_batch))

    corpus, id_to_word, word_to_id = create_corpus_and_dict(text_list)
    save_data(pickle_name, (corpus, id_to_word, word_to_id))
    return corpus, id_to_word, word_to_id