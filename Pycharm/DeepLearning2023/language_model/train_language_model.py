from language_model import LanguageModel
from common.trainer import Trainer
from common.utils import py, get_processed_essay_data, create_essay_corpus_and_dict, get_one_hot_encoding
from common.adam_optimizer import AdamOptimizer

if __name__ == '__main__':

    corpus, id_to_word, word_to_id = create_essay_corpus_and_dict()
    essay_data_list = get_processed_essay_data(load_test_data=False,
                                               word_to_id=word_to_id,
                                               load_pickle=True)
    essay_data_list = essay_data_list[:10000]

    vocab_size = len(id_to_word)
    wordvec_size = 100
    hidden_size = 1000

    model = LanguageModel(vocab_size=vocab_size,
                          wordvec_size=wordvec_size,
                          hidden_size=hidden_size)

    optimizer = AdamOptimizer()
    trainer = Trainer(model, optimizer)

    for essay_data in essay_data_list:
        paragraphs = essay_data['paragraph']

        for paragraph in paragraphs:
            xs = paragraph[:1]
            ts = paragraph[1:]
            trainer.fit(xs, ts, batch_size=1, random_batch=False, max_epoch=10)
            trainer.plot()




