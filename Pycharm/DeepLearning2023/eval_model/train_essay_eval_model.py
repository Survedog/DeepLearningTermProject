from eval_model.essay_eval_model import EssayEvalModel
from lang_model.language_model import LanguageModel
from common.utils import py, create_essay_corpus_and_dict, get_processed_essay_data
from common.trainer import Trainer
from common.adam_optimizer import AdamOptimizer

if __name__ == '__main__':

    _, id_to_word, word_to_id = create_essay_corpus_and_dict(load_pickle=True)
    train_data_list = get_processed_essay_data(load_test_data=False, word_to_id=word_to_id, load_pickle=True)

    vocab_size = len(id_to_word)
    wordvec_size = 100
    hidden_size = 100
    time_size = 30

    rnn_model = LanguageModel(vocab_size, wordvec_size, hidden_size)
    rnn_model.load_params()

    eval_model = EssayEvalModel(rnn_model, time_size,
                                fit_premade_models=False)
    optimizer = AdamOptimizer()
    trainer = Trainer(eval_model, optimizer)

    x_t_list = EssayEvalModel.get_x_t_list_from_processed_data(train_data_list[:10], time_size=time_size, load_pickle=False, save_pickle=True)
    for (x, t) in x_t_list:
        eval_model.forward(x, t)
        # trainer.fit(x, t)
