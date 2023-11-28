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

    id_count = 1
    for word in parsed_text:
        if word not in word_to_id:
            id_to_word[id_count] = word
            word_to_id[word] = id_count
            id_count += 1

        corpus.append(word_to_id[word])

    return corpus, id_to_word, word_to_id



