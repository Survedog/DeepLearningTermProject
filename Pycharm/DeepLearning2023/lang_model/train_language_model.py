from lang_model.language_model import LanguageModel
from common.trainer import RnnlmTrainer
from common.utils import py, get_processed_essay_data, create_essay_corpus_and_dict
from common.adam_optimizer import AdamOptimizer
import random

if __name__ == '__main__':

    _, id_to_word, word_to_id = create_essay_corpus_and_dict()
    train_data_list = get_processed_essay_data(load_test_data=False,
                                               word_to_id=word_to_id,
                                               load_pickle=True)
    train_data_list = train_data_list[:10]

    vocab_size = len(id_to_word)
    wordvec_size = 100
    hidden_size = 100
    time_size = 30
    batch_size = 10

    model = LanguageModel(vocab_size=vocab_size,
                          wordvec_size=wordvec_size,
                          hidden_size=hidden_size)

    optimizer = AdamOptimizer()
    trainer = RnnlmTrainer(model, optimizer)

    # 학습
    for test_data in train_data_list:
        essay_corpus = py.array(sum(test_data['paragraph'], []))
        xs = essay_corpus[:-1]
        ts = essay_corpus[1:]

        trainer.fit(xs, ts,
                    time_size=min(time_size, len(xs)),
                    batch_size=batch_size,
                    max_epoch=10)

    # 평가
    test_data_list = get_processed_essay_data(load_test_data=True,
                                              word_to_id=word_to_id,
                                              load_pickle=True)
    random.shuffle(test_data_list)
    test_data_list = test_data_list[:1]

    total_loss = 0
    for test_data in test_data_list:
        essay_corpus = sum(test_data['paragraph'], [])
        xs = essay_corpus[:-1]
        ts = essay_corpus[1:]

        model.reset_state()
        total_loss += model.forward(xs, ts)

    perplexity = py.exp(total_loss / len(test_data_list))
    print('테스트 perplexity: %0.2f' % perplexity)

