from language_model import LanguageModel
from common.trainer import Trainer
from common.utils import load_essay_data_list

if __name__ == '__main__':

    vocab_size = 10000
    wordvec_size = 1000
    hidden_size = 1000

    model = LanguageModel(vocab_size=vocab_size,
                          wordvec_size=wordvec_size,
                          hidden_size=hidden_size)

    train_data = load_essay_data_list(load_test_data=False)
