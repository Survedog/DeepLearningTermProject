from eval_model.essay_eval_model import EssayEvalModel
from common.utils import py, create_essay_corpus_and_dict, get_processed_essay_data, load_data
from common.trainer import EssayEvalModelTrainer
from common.adam_optimizer import AdamOptimizer
import random

if __name__ == '__main__':

    print('Loading corpus...')
    _, id_to_word, word_to_id = create_essay_corpus_and_dict(load_pickle=True)

    if '<UNK>' not in word_to_id:
        unknown_id = len(word_to_id)
        id_to_word[unknown_id] = '<UNK>'
        word_to_id['<UNK>'] = unknown_id

    train_data_list = get_processed_essay_data(load_test_data=False, word_to_id=word_to_id, load_pickle=True)
    embed_weight, _ = load_data('cbow_params.p')

    do_fitting = False  # 학습을 수행할 지
    load_saved_param = True  # 모델의 저장된 매개변수(가중치)를 불러올 지
    save_param = False  # 학습 후 매개변수를 저장할 지

    vocab_size = len(id_to_word)
    wordvec_size = embed_weight.shape[-1]
    lstm_hidden_size = 100
    time_size = 80

    train_data_start = 0  # 학습을 데이터의 어디서부터 시작할지
    train_data_size = 10000  # 학습 데이터 크기
    test_data_size = 1000  # 테스트 데이터 크기
    max_epoch = 2

    print('Creating model...')
    eval_model = EssayEvalModel(vocab_size, wordvec_size, lstm_hidden_size, time_size, embed_weight=embed_weight)
    optimizer = AdamOptimizer()
    trainer = EssayEvalModelTrainer(eval_model, optimizer)

    if load_saved_param:
        eval_model.load_params()

    # 학습
    if do_fitting:
        print('Loading train data...')
        x_list, t_list = EssayEvalModel.get_x_t_list_from_processed_data(train_data_list[train_data_start: train_data_start + train_data_size], time_size=time_size, load_pickle=False, save_pickle=True)
        print('Start fitting...')
        trainer.fit(x_list, t_list,
                    max_epoch=max_epoch)
        trainer.plot()

        if save_param:
            eval_model.save_params()

    # 평가
    test_data_list = get_processed_essay_data(load_test_data=True, word_to_id=word_to_id, load_pickle=True)
    random.shuffle(test_data_list)
    x_list, t_list = EssayEvalModel.get_x_t_list_from_processed_data(test_data_list[:test_data_size], time_size=time_size, load_pickle=False, save_pickle=False)

    total_loss = 0
    loss_count = 0
    test_id = 0

    for x, t in zip(x_list, t_list):
        test_id += 1
        eval_model.reset_state()
        loss = eval_model.forward(x, t, train_flag=False)
        print('Test %d - Loss: %.2f' % (test_id, loss))

        total_loss += loss
        loss_count += 1

    print('Final Test Loss: %.2f' % (total_loss / loss_count))

    # 예측 값
    predict_count = 1000

    total_diff = 0
    for x, t in zip(x_list[:predict_count], t_list[:predict_count]):
        eval_model.reset_state()
        prediction = eval_model.predict(x, train_flag=False)
        total_diff += prediction - t

    print('[평균 오차]\n', (total_diff / predict_count))
    print('[오차 절대값의 전체 평균] %.2f\n' % py.abs(total_diff / predict_count).mean())
    print('[오차 절대값 합계] %.2f\n' % py.abs(total_diff / predict_count).sum())
