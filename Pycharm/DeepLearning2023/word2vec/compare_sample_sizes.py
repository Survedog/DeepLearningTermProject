from common.utils import py, create_essay_corpus_and_dict, create_test_essay_corpus, create_context_and_target
from word2vec.cbow_model import CBowModel
from common.adam_optimizer import AdamOptimizer
from common.trainer import Trainer
from config import Config
import time
from matplotlib import pyplot as plt

if __name__ == '__main__':

    print('Creating Corpus...')
    corpus, id_to_word, word_to_id = create_essay_corpus_and_dict(load_pickle=True)
    test_corpus = create_test_essay_corpus(word_to_id, load_pickle=True, save_pickle=True)

    if Config.USE_GPU:
        if not isinstance(corpus, py.ndarray):
            corpus = py.array(corpus)

    print('Creating Context and targets...')
    train_context, train_target = create_context_and_target(corpus[1000000:2000000])
    test_context, test_target = create_context_and_target(test_corpus[10000:20000])

    # word2vec 설정
    hidden_size = 50
    vocab_size = len(id_to_word)
    sample_size_list = [5, 10, 100, 500, 1000]

    # 옵티마이저 설정
    learning_rate = 0.001

    # 트레이너 설정
    max_epoch = 5
    batch_size = 1000
    load_saved_params = False

    # plot
    test_loss = []
    train_loss = []
    train_time = []

    for sample_size in sample_size_list:
        print('[Sample Size %d]\nCreating model...' % sample_size)
        weight_in = py.random.randn(vocab_size, hidden_size)
        weight_out = py.random.randn(vocab_size, hidden_size)

        model = CBowModel(corpus, vocab_size, hidden_size, sample_size, weight_in, weight_out)
        optimizer = AdamOptimizer(learning_rate)
        trainer = Trainer(model, optimizer)

        if load_saved_params:
            print('Loading params...')
            model.load_params()

        # 학습
        start = time.time()
        print('Fitting model...')
        trainer.fit(x=train_context,
                    t=train_target,
                    batch_size=batch_size,
                    max_epoch=max_epoch,
                    eval_interval=1)
        train_time.append(time.time() - start)
        train_loss.append(float(trainer.loss_list[-1]))

        # 평가
        t_loss = model.forward(test_context, test_target).mean()
        test_loss.append(float(t_loss))
        print('Sample Size %d - Test Loss: %.2f' % (sample_size, t_loss))

    plt.subplot(2, 1, 1)
    plt.plot(sample_size_list, train_loss, 'r-', label='train loss')
    plt.plot(sample_size_list, test_loss, 'b-', label='test loss')
    plt.title('CBOW sample size compare')
    plt.xlabel('Sample size')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(sample_size_list, train_time, 'g--', label='time')
    plt.title('CBOW sample size compare')
    plt.xlabel('Sample size')
    plt.ylabel('Time (second)')
    plt.legend()

    plt.tight_layout()
    plt.show()

