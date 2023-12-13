import numpy
from common.utils import py, create_essay_corpus_and_dict, create_test_essay_corpus, create_context_and_target
from word2vec.cbow_model import CBowModel
from common.adam_optimizer import AdamOptimizer
from common.trainer import Trainer
from config import Config

if __name__ == '__main__':

    print('Creating Corpus...')
    corpus, id_to_word, word_to_id = create_essay_corpus_and_dict(load_pickle=True)
    if Config.USE_GPU:
        if not isinstance(corpus, py.ndarray):
            corpus = py.array(corpus)

    print('Creating Context and targets...')
    context, target = create_context_and_target(corpus[:10000000])

    # word2vec 설정
    hidden_size = 50
    sample_size = 100
    vocab_size = len(id_to_word)
    weight_in = py.random.randn(vocab_size, hidden_size)
    weight_out = py.random.randn(vocab_size, hidden_size)

    # 옵티마이저 설정
    learning_rate = 0.001

    # 트레이너 설정
    max_epoch = 5
    batch_size = 10000
    do_fitting = False
    continue_from_last_fit = True
    save_params = False

    print('Creating model...')
    model = CBowModel(corpus, vocab_size, hidden_size, sample_size, weight_in, weight_out)
    optimizer = AdamOptimizer(learning_rate)
    trainer = Trainer(model, optimizer)

    if continue_from_last_fit or not do_fitting:
        print('Loading params...')
        model.load_params()

    # 학습
    if do_fitting:
        print('Fitting model...')
        trainer.fit(x=context,
                    t=target,
                    batch_size=batch_size,
                    max_epoch=max_epoch,
                    eval_interval=1)
        trainer.plot()

        if save_params:
            print('Saving params...')
            model.save_params()

    # 평가
    test_size = 10000
    test_corpus = create_test_essay_corpus(word_to_id, load_pickle=True, save_pickle=True)

    context, target = create_context_and_target(test_corpus[:test_size])
    test_loss = model.forward(context, target).mean()
    print('Test Loss: %.2f' % test_loss)

    # 예측
    eval_max_iter = 100
    total_correct_count = 0
    question_per_iter = 100

    for _ in range(eval_max_iter):
        rand_idx = py.random.permutation(py.arange(question_per_iter))
        predict_context = context[rand_idx]
        predict_target = target[rand_idx]
        prediction = model.predict(predict_context)

        if Config.USE_GPU:
            predict_context = py.asnumpy(predict_context)
            predict_target = py.asnumpy(predict_target)
            prediction = py.asnumpy(prediction)

        correct_count = 0
        for i in range(len(predict_context)):
            if numpy.array_equal(predict_target[i], prediction[i]):
                correct_count += 1
            print('문맥: %s/%s\t\t| 예측/정답: %s/%s' %
                  (id_to_word[predict_context[i, 0]], id_to_word[predict_context[i, 1]],
                   id_to_word[prediction[i]], id_to_word[predict_target[i]]))

        print('정답율: %.1f' % (correct_count * 100 / question_per_iter))
        total_correct_count += correct_count

    print('최종 정답율: %.1f' % (total_correct_count * 100 / (question_per_iter * eval_max_iter)))
