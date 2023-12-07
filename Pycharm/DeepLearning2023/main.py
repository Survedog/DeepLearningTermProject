from utils import py, create_essay_corpus_and_dict, create_context_and_target
from word2vec.cbow_layer import CBowLayer
from adam_optimizer import AdamOptimizer
from trainer import Trainer
from config import Config

if __name__ == '__main__':

    print('Creating Corpus...')
    corpus, id_to_word, word_to_id = create_essay_corpus_and_dict(load_pickle=True)
    if Config.USE_GPU:
        if not isinstance(corpus, py.ndarray):
            corpus = py.array(corpus)

    print('Creating Context and targets...')
    context, target = create_context_and_target(corpus[:1000000])

    # word2vec 설정
    hidden_size = 100
    sample_size = 1000
    vocab_size = len(id_to_word)
    weight_in = py.random.rand(vocab_size, hidden_size)
    weight_out = py.random.rand(vocab_size, hidden_size)

    # 옵티마이저 설정
    learning_rate = 0.001

    # 트레이너 설정
    max_epoch = 20
    batch_size = 1000
    should_fit = False

    print('Creating model...')
    word2vec = CBowLayer(corpus, vocab_size, hidden_size, sample_size, weight_in, weight_out)
    optimizer = AdamOptimizer(learning_rate)
    trainer = Trainer(word2vec, optimizer)

    if should_fit:
        print('Fitting model...')
        trainer.fit(train_data=context,
                    answer=target,
                    batch_size=batch_size,
                    max_epoch=max_epoch,
                    print_interval=1)

        print('Saving params...')
        word2vec.save_params()
        trainer.plot_loss()
    else:
        print('Loading params...')
        word2vec.load_params()

    total_count = 100
    correct_count = 0

    rand_idx = py.random.permutation(py.arange(total_count))
    predict_context = context[rand_idx]
    predict_target = target[rand_idx]
    prediction = word2vec.predict(predict_context)

    if Config.USE_GPU:
        predict_context = py.asnumpy(predict_context)
        predict_target = py.asnumpy(predict_target)
        prediction = py.asnumpy(prediction)

    for i in range(len(predict_context)):
        if (predict_target[i] == prediction[i]):
            correct_count += 1
        print('문맥: %s/%s\t\t| 예측/정답: %s/%s' %
              (id_to_word[predict_context[i, 0]], id_to_word[predict_context[i, 1]],
               id_to_word[prediction[i]], id_to_word[predict_target[i]]))

    print('정답율: %.1f' % (correct_count * 100 / total_count))
