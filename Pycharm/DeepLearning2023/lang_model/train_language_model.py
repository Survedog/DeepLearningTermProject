from lang_model.language_model import LanguageModel
from common.trainer import RnnlmTrainer
from common.utils import py, get_processed_essay_data, create_essay_corpus_and_dict
from common.adam_optimizer import AdamOptimizer
import random

if __name__ == '__main__':

    print('Loading corpus...')
    _, id_to_word, word_to_id = create_essay_corpus_and_dict(load_pickle=True)

    do_fitting = False
    load_saved_param = True
    save_param = False

    vocab_size = len(id_to_word)
    wordvec_size = 100
    hidden_size = 100
    time_size = 30
    batch_size = 10

    print('Creating model...')
    model = LanguageModel(vocab_size=vocab_size,
                          wordvec_size=wordvec_size,
                          hidden_size=hidden_size)

    optimizer = AdamOptimizer()
    trainer = RnnlmTrainer(model, optimizer)

    if load_saved_param:
        model.load_params()

    # 학습
    if do_fitting:
        print('Loading train data...')
        train_data_list = get_processed_essay_data(load_test_data=False,
                                                   word_to_id=word_to_id,
                                                   load_pickle=True)
        train_data_list = train_data_list[1000:3000]
        train_count = 0

        for train_data in train_data_list:
            train_count += 1

            essay_corpus = py.array(sum(train_data['paragraph'], []))
            xs = essay_corpus[:-1]
            ts = essay_corpus[1:]

            trainer.fit(xs, ts,
                        time_size=min(time_size, len(xs)),
                        batch_size=batch_size,
                        max_epoch=10)
            print('%d번 데이터 학습 완료.' % train_count)

        if save_param:
            model.save_params()

    # 평가
    test_time_size = 50

    print('Loading test data...')
    test_data_list = get_processed_essay_data(load_test_data=True,
                                              word_to_id=word_to_id,
                                              load_pickle=True)
    random.shuffle(test_data_list)
    test_data_list = test_data_list[:100]

    test_id = 0
    total_loss = 0
    for test_data in test_data_list:
        test_id += 1
        essay_corpus = py.array(sum(test_data['paragraph'], []))
        test_len = test_time_size * (len(essay_corpus) // test_time_size)

        essay_corpus = essay_corpus[:test_len].reshape(-1, test_time_size)
        xs = essay_corpus[:, :-1]
        ts = essay_corpus[:, 1:]

        model.reset_state()
        loss = model.forward(xs, ts)
        total_loss += loss
        print('Test %d - perplexity: %0.2f' % (test_id, py.exp(loss)))

    perplexity = py.exp(total_loss / len(test_data_list))
    print('Final Test perplexity: %0.2f' % perplexity)
