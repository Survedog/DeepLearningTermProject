import random
import traceback
from config import Config
from konlpy.tag import Komoran, Kkma
from pathlib import Path
import json
import pickle
import numpy

if Config.USE_GPU:
    import cupy as py
    py.cuda.set_allocator(py.cuda.MemoryPool().malloc)
else:
    import numpy as py

text_parser = Komoran()
# text_parser = Kkma()

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

    unknown_token_id = len(word_to_id)
    word_to_id['<UNK>'] = unknown_token_id
    id_to_word[unknown_token_id] = '<UNK>'

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
        encoded = py.zeros(array_size, dtype=int)
        encoded[num] = 1
    else:
        encoded = py.zeros(num.shape + (array_size,), dtype=int)

        # 1이 되어야 하는 각 원소에 접근하기 위한 인덱스를 생성한다.
        index = get_ndarray_index(num) + tuple([num.flatten()])
        encoded[index] = 1
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

    index = get_ndarray_index(t)
    batch_index = index[0].reshape(t.shape)
    sample_index = index[1].reshape(t.shape)
    return (-py.log(y[batch_index, sample_index, t])).reshape(original_t_shape)


def get_sum_of_squares_error(y, t):
    sse = py.square(y - t).sum(axis=-1)
    sse /= 2  # delta rule
    return sse


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


def create_essay_corpus_and_dict(load_pickle=True, save_pickle=True, batch_size=100):
    pickle_name = 'corpus_and_dicts.p'
    if load_pickle:
        data = load_data(pickle_name)
        if data is not None:
            return data

    essay_data_list = load_essay_data_list(load_test_data=False, load_pickle=True)
    text_list = get_joined_paragraph_text_list(essay_data_list, batch_size)
    corpus, id_to_word, word_to_id = create_corpus_and_dict(text_list)

    if save_pickle:
        save_data(pickle_name, (corpus, id_to_word, word_to_id))
    return corpus, id_to_word, word_to_id


def create_test_essay_corpus(word_to_id, load_pickle=True, save_pickle=True):
    pickle_name = 'essay_corpus_test.p'
    if load_pickle:
        data = load_data(pickle_name)
        if data is not None:
            return data

    test_essay_data_list = load_essay_data_list(load_test_data=True, load_pickle=True)
    test_corpus = text_to_ids(test_essay_data_list, word_to_id)

    if save_pickle:
        save_data(pickle_name, test_corpus)
    return test_corpus


def get_joined_paragraph_text_list(essay_data_list, batch_size=100):
    """
    :param essay_data_list: 에세이 json 파일을 불러온 딕셔너리 리스트
    :param batch_size: 몇 개의 paragraph마다 하나로 묶을지
    :return: paragraph 문자열을 batch_size개씩 연결한 문자열들의 리스트
    """
    text_list, text_batch = [], []

    for essay_data in essay_data_list:
        for paragraph in essay_data['paragraph']:
            text = paragraph['paragraph_txt'].replace('\n', '').replace('\r', '').replace('\t', '')
            text_batch += text.split('#@문장구분#')[0:-1]

            if len(text_batch) >= batch_size:
                text_list.append(''.join(text_batch[:batch_size]))
                text_batch = text_batch[batch_size:]

    text_list.append(''.join(text_batch))

def text_to_ids(text_list, word_to_id):
    word_ids = []
    unknown_token_id = word_to_id.get('<UNK>', len(word_to_id))

    for i, text in enumerate(text_list):
        print('Converting text to ids[%d/%d]' % (i, len(text_list)))
        try:
            parsed_text = text_parser.morphs(text)
        except UnicodeDecodeError:
            print('Error while parsing text[%d/%d]' % (i, len(text_list)))
            continue

        for word in parsed_text:
            word_ids.append(word_to_id.get(word, unknown_token_id))

    return py.array(word_ids)

def get_processed_essay_data(load_test_data, word_to_id, load_pickle=False, max_count=None, shuffle=False):
    pickle_name = 'processed_essay_data_' + ('test.p' if load_test_data else 'train.p')
    if load_pickle:
        eval_data_list = load_data(pickle_name)
        if eval_data_list is not None:
            return eval_data_list

    eval_data_list = []
    raw_data_list = load_essay_data_list(load_test_data=load_test_data)
    essay_type = {'글짓기': 0, '설명글': 0, '주장': 1, '찬성반대': 1, '대안제시': 1}
    unknown_token_id = len(word_to_id) - 1

    if shuffle:
        random.shuffle(raw_data_list)
    if max_count is not None:
        raw_data_list = raw_data_list[:max_count]

    data_num = 0
    for raw_data in raw_data_list:
        data_num += 1
        print('[DATA PROCESS] processing essay data no.%d...' % data_num)

        eval_data = {}
        eval_data['type'] = essay_type[raw_data['info']['essay_type']]
        eval_data['corr_count'] = len(raw_data['correction'])

        score_detail = raw_data['score']['essay_scoreT_detail']
        eval_data['score'] = {'org': score_detail['essay_scoreT_org'],
                              'cont': score_detail['essay_scoreT_cont'],
                              'exp': score_detail['essay_scoreT_exp']}

        eval_data['weight'] = {'org': raw_data['rubric']['organization_weight'],
                               'cont': raw_data['rubric']['content_weight'],
                               'exp': raw_data['rubric']['expression_weight']}

        eval_data['paragraph'] = []
        for paragraph in raw_data['paragraph']:
            #todo: #@문장구분# 처리
            words = paragraph['paragraph_txt'].replace('\n', '').replace('\r', '').replace('\t', '')
            try:
                parsed_words = text_parser.morphs(words)
            except Exception:
                print('[DATA PROCESS] error parsing data %d.\n%s' % (data_num, traceback.format_exc()))
                eval_data['paragraph'].append(None)
                continue

            word_ids = []
            for parsed_word in parsed_words:
                word_ids.append(word_to_id.get(parsed_word, unknown_token_id))
            eval_data['paragraph'].append(word_ids)

        eval_data_list.append(eval_data)

    save_data(pickle_name, eval_data_list)
    return eval_data_list

def get_ndarray_index(num):
    """
    :param num: numpy/cupy 배열
    :return: 각 원소에 접근하기 위한 각 차원의 인덱스들의 튜플
    ex) num = py.array([[5, 4, 3], [3, 4, 5]])
        return = ([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])
    """
    index = []
    left = 1  # 각 차원의 상위 차원 개수
    for s in num.shape:
        right = num.size // (s * left)  # 각 차원의 하위 차원 개수
        dim_index = py.tile(py.repeat(py.arange(s), right), left)
        index.append(dim_index)
        left *= s
    return tuple(index)
