import numpy
from common.utils import py, create_essay_corpus_and_dict, create_context_and_target
from word2vec.skipgram_model import SkipgramModel
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
    target, context = create_context_and_target(corpus[:10000000])

    # word2vec 설정
    wordvec_size = 100
    sample_size = 50
    vocab_size = len(id_to_word)
    weight_in = py.random.rand(vocab_size, wordvec_size)
    weight_out_list = [py.random.rand(vocab_size, wordvec_size) for _ in range(target.shape[-1])]

    # 옵티마이저 설정
    learning_rate = 0.001

    # 트레이너 설정
    max_epoch = 8
    batch_size = 30000
    do_fitting = True
    continue_from_last_fit = True
    save_params = True

    print('Creating model...')
    model = SkipgramModel(corpus=corpus,
                          vocab_size=vocab_size,
                          wordvec_size=wordvec_size,
                          sample_size=sample_size,
                          weight_in=weight_in,
                          weight_out_list=weight_out_list)
    optimizer = AdamOptimizer(learning_rate)
    trainer = Trainer(model, optimizer)

    if continue_from_last_fit or not do_fitting:
        print('Loading params...')
        model.load_params()

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

    total_count = 100
    correct_count = 0

    rand_idx = py.random.permutation(py.arange(total_count))
    predict_context = context[rand_idx]
    predict_target = target[rand_idx]
    prediction = model.predict(predict_context)

    if Config.USE_GPU:
        predict_context = py.asnumpy(predict_context)
        predict_target = py.asnumpy(predict_target)
        prediction = py.asnumpy(prediction)

    for i in range(len(predict_context)):
        if numpy.array_equal(predict_target[i], prediction[i]):
            correct_count += 1
        print('문맥: %s\t\t| 예측/정답: (%s, %s)/(%s, %s)' %
              (id_to_word[predict_context[i]],
               id_to_word[prediction[i, 0]], id_to_word[prediction[i, 1]],
               id_to_word[predict_target[i, 0]], id_to_word[predict_target[i, 1]]))

    print('정답율: %.1f' % (correct_count * 100 / total_count))
