from eval_model.essay_eval_model import EssayEvalModel
from lang_model.time_rnn_layers import TimeLSTMLayer
from common.utils import py, create_essay_corpus_and_dict, get_processed_essay_data
from common.trainer import Trainer
from common.adam_optimizer import AdamOptimizer

if __name__ == '__main__':

    print('Loading corpus...')
    _, id_to_word, word_to_id = create_essay_corpus_and_dict(load_pickle=True)
    train_data_list = get_processed_essay_data(load_test_data=False, word_to_id=word_to_id, load_pickle=True)

    do_fitting = True
    load_saved_param = False
    save_param = True

    vocab_size = len(id_to_word)
    wordvec_size = 100
    hidden_size = 100
    time_size = 30

    print('Creating model...')
    eval_model = EssayEvalModel(vocab_size, wordvec_size, hidden_size, time_size)
    optimizer = AdamOptimizer()
    trainer = Trainer(eval_model, optimizer)

    if load_saved_param:
        eval_model.load_params()

    if do_fitting:
        print('Loading train data...')
        x_t_list = EssayEvalModel.get_x_t_list_from_processed_data(train_data_list[:10], time_size=time_size, load_pickle=False, save_pickle=True)
        train_count = 0

        for (x, t) in x_t_list:
            train_count += 1
            trainer.fit(x, t)
            print('%d번 데이터 학습 완료.' % train_count)

        if save_param:
            eval_model.save_params()
