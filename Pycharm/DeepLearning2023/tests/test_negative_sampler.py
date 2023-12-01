import unittest
from negative_sampler import *


class EmbeddingLayerTests(unittest.TestCase):

    def test_get_negative_samples(self):
        sample_size = 1000
        value_space = np.array([0, 1, 2, 3, 4])
        distribution = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        sampler = NegativeSampler(value_space, distribution)

        # 샘플에서 positive 값이 제외되는지 확인
        for i in range(100):
            positive_idx = np.array([i % 5])
            negative_samples = sampler.get_negative_samples(sample_size, positive_idx)
            self.assertNotIn(positive_idx, negative_samples)

            for sample in negative_samples[0]:
                self.assertIn(sample, value_space)

        # 배치 처리 확인
        positive_idx_list = np.array([0, 1, 2, 3])
        negative_samples = sampler.get_negative_samples(sample_size, positive_idx_list)

        for i, positive_idx in enumerate(positive_idx_list):
            self.assertNotIn(positive_idx, negative_samples[i])
            for sample in negative_samples[i]:
                self.assertIn(sample, value_space)
