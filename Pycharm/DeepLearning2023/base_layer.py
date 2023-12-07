from abc import *
from utils import load_data, save_data


class LayerBase(metaclass=ABCMeta):

    def __init__(self):
        self.params = []
        self.grads = []
        self.default_params_pickle_name = 'default.p'
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    def save_params(self, pickle_name=None):
        if pickle_name is None:
            pickle_name = self.default_params_pickle_name
        save_data(pickle_name, self.params)

    def load_params(self, pickle_name=None):
        if pickle_name is None:
            pickle_name = self.default_params_pickle_name
        data = load_data(pickle_name)
        self.params = data if (data is not None) else []
