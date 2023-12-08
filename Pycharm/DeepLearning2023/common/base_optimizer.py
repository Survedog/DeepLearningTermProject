from abc import *


class OptimizerBase(metaclass=ABCMeta):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, params, grads):
        pass
