import unittest
from utils import py, create_corpus_and_dict, create_context_and_target
from word2vec.cbow_model import CBowModel
from adam_optimizer import AdamOptimizer
from trainer import Trainer
from config import Config


class TrainerTests(unittest.TestCase):

    def test_fit(self):
        train_text = '안녕하세요. 만나서 반갑습니다. 저는 프로그래머입니다.'
        corpus, id_to_word, word_to_id = create_corpus_and_dict(train_text)
        context, target = create_context_and_target(corpus)

        # word2vec 설정
        hidden_size = 5
        sample_size = 10
        vocab_size = len(id_to_word)
        weight_in = py.random.rand(vocab_size, hidden_size)
        weight_out = py.random.rand(vocab_size, hidden_size)

        # 옵티마이저 설정
        learning_rate = 0.001

        # 트레이너 설정
        max_epoch = 1000
        batch_size = 10

        word2vec = CBowModel(corpus, vocab_size, hidden_size, sample_size, weight_in, weight_out)
        optimizer = AdamOptimizer(learning_rate)
        trainer = Trainer(word2vec, optimizer)

        trainer.fit(train_data=context,
                    answer=target,
                    batch_size=batch_size,
                    max_epoch=max_epoch,
                    print_interval=100)

        prediction = word2vec.predict(context)

        if Config.USE_GPU:
            prediction = py.asnumpy(prediction)
            context = py.asnumpy(context)
            target = py.asnumpy(target)

        for i in range(len(context)):
            print('[문맥]: (%s, %s)\t\t| [예측/정답]: %s/%s' %
                  (id_to_word[context[i, 0]], id_to_word[context[i, 1]],
                   id_to_word[prediction[i]], id_to_word[target[i]]))
