from config import Config

if Config.USE_GPU:
    import cupy as np
else:
    import numpy as np


def sigmoid(values):
    return 1 / (1 + np.exp(-values))





