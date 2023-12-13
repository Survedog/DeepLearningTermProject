from eval_model.essay_eval_model import EssayEvalModel
from eval_model.essay_eval_model_2 import EssayEvalModel2
from common.utils import create_essay_corpus_and_dict, get_processed_essay_data, load_data
from common.trainer import EssayEvalModelTrainer
from common.adam_optimizer import AdamOptimizer
from matplotlib import pyplot as plt
import random

if __name__ == '__main__':

    print('Loading corpus...')
    _, id_to_word, word_to_id = create_essay_corpus_and_dict(load_pickle=True)

    if '<UNK>' not in word_to_id:
        unknown_id = len(word_to_id)
        id_to_word[unknown_id] = '<UNK>'
        word_to_id['<UNK>'] = unknown_id

    train_data_list = get_processed_essay_data(load_test_data=False, word_to_id=word_to_id, load_pickle=True)
    test_data_list = get_processed_essay_data(load_test_data=True, word_to_id=word_to_id, load_pickle=True)
    random.shuffle(test_data_list)

    # 모델 설정
    embed_weight, _ = load_data('cbow_params.p')
    vocab_size = len(id_to_word)
    wordvec_size = embed_weight.shape[-1]
    lstm1_hidden_size = 100
    lstm2_hidden_size = 11
    time_size = 80

    print('Creating model...')
    eval_model1 = EssayEvalModel(vocab_size, wordvec_size, lstm1_hidden_size, time_size, embed_weight=embed_weight,
                                 dropout_rate=0.5)
    eval_model2 = EssayEvalModel2(vocab_size, wordvec_size, lstm1_hidden_size, lstm2_hidden_size, time_size,
                                  embed_weight=embed_weight, dropout_rate=0.3)
    optimizer1 = AdamOptimizer()
    optimizer2 = AdamOptimizer()
    trainer1 = EssayEvalModelTrainer(eval_model1, optimizer1)
    trainer2 = EssayEvalModelTrainer(eval_model2, optimizer2)

    # 학습 설정
    max_epoch = 3
    train_data_size = 10000
    test_data_size = 1000
    max_compare_count = 4

    train_per_compare = train_data_size // max_compare_count
    test_per_compare = test_data_size // max_compare_count
    save_param = True

    print('Loading train/test data...')
    x_train, t_train = EssayEvalModel.get_x_t_list_from_processed_data(train_data_list[: train_data_size],
                                                                       time_size=time_size, load_pickle=False,
                                                                       save_pickle=False)
    x_test, t_test = EssayEvalModel.get_x_t_list_from_processed_data(test_data_list[:test_data_size],
                                                                     time_size=time_size, load_pickle=False,
                                                                     save_pickle=False)

    # plot
    test_plot_x = []
    test_plot_y1 = []
    test_plot_y2 = []

    train_from, test_from = 0, 0
    for train_count in range(max_compare_count):
        train_end = train_from + train_per_compare
        test_end = test_from + test_per_compare

        print('Start fitting...')
        trainer1.fit(x_train[train_from:train_end], t_train[train_from:train_end], max_epoch=max_epoch)
        trainer2.fit(x_train[train_from:train_end], t_train[train_from:train_end], max_epoch=max_epoch)

        if save_param:
            eval_model1.save_params()
            eval_model2.save_params()

        print('Start testing...')
        total_loss1, total_loss2 = 0.0, 0.0
        loss_count = 0
        for x, t in zip(x_test[test_from:test_end], t_test[test_from:test_end]):
            loss_count += 1
            eval_model1.reset_state()
            eval_model2.reset_state()
            total_loss1 += eval_model1.forward(x, t, train_flag=False)
            total_loss2 += eval_model2.forward(x, t, train_flag=False)
        print('Test - Loss1: %.2f, Loss2: %.2f' % (total_loss1/loss_count, total_loss2/loss_count))

        test_plot_x.append(train_from)
        test_plot_y1.append(float(total_loss1 / loss_count))
        test_plot_y2.append(float(total_loss2 / loss_count))

        train_from += train_per_compare
        test_from += test_per_compare

    plt.plot(test_plot_x, test_plot_y1, 'r-', label='EEM1')
    plt.plot(test_plot_x, test_plot_y2, 'b-', label='EEM2')

    plt.xlabel('Train data size')
    plt.ylabel('Test loss')
    plt.title('EEM Compare')

    plt.legend()
    plt.show()
