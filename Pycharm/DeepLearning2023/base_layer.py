from abc import *


class LayerBase(metaclass=ABCMeta):

    def __init__(self):
        self.param = []
        self.grads = []
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass
