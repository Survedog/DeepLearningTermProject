from abc import *


class LayerBase(metaclass=ABCMeta):

    def __init__(self):
        self.param = None
        self.grads = None
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass
