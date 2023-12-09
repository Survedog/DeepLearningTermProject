from common.utils import create_essay_corpus_and_dict, get_processed_essay_data
from eval_model.essay_eval_model import EssayEvalModel

if __name__ == '__main__':

    corpus, id_to_word, word_to_id = create_essay_corpus_and_dict()
    if '<UNK>' not in word_to_id:
        unknown_id = len(word_to_id)
        id_to_word[unknown_id] = '<UNK>'
        word_to_id['<UNK>'] = unknown_id

    essay_data = get_processed_essay_data(load_test_data=False, word_to_id=word_to_id, max_count=10000)
    model = EssayEvalModel()
