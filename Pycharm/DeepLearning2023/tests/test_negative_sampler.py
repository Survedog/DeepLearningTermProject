import unittest
from negative_sampler import *


class EmbeddingLayerTests(unittest.TestCase):

    def test_get_negative_samples(self):
        sample_size = 1000
        value_space = np.array([0, 1, 2, 3, 4])
        distribution = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        sampler = NegativeSampler(sample_size, value_space, distribution)

        for i in range(100):
            positive = i % 5
            negative_samples = sampler.get_negative_samples(positive)
            self.assertNotIn(positive, negative_samples)

            for sample in negative_samples:
                self.assertIn(sample, value_space)
