from config import Config

if Config.USE_GPU:
    import cupy as np
else:
    import numpy as np


class NegativeSampler:

    def __init__(self, sample_size, value_space, distribution):
        self.sample_size = sample_size
        self.value_space = value_space
        self.distribution = distribution

    def get_negative_samples(self, positive_idx):
        p_positive = self.distribution[positive_idx]
        self.distribution[positive_idx] = 0.0

        negative_samples = np.random.choice(self.value_space,
                                            self.sample_size,
                                            replace=True,
                                            p=self.distribution / (1 - p_positive))

        self.distribution[positive_idx] = p_positive
        return negative_samples
