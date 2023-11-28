from base_layer import LayerBase
from config import Config

if Config.USE_GPU:
    import cupy as np
else:
    import numpy as np


class SkipGram(LayerBase):

    def __init__(self):
        super().__init__()

        pass

    def forward(self):
        pass

    def backward(self):
        pass
