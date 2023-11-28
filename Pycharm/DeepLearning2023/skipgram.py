from config import Config

if Config.USE_GPU:
    import cupy as np
else:
    import numpy as np


