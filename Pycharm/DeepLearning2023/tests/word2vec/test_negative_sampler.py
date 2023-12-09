import unittest
from word2vec.negative_sampler import *


class EmbeddingLayerTests(unittest.TestCase):

    def test_get_negative_samples(self):
        sample_size = 1000
        value_space = py.array([0, 1, 2, 3, 4])
        distribution = py.array([0.2, 0.2, 0.2, 0.2, 0.2])
        sampler = NegativeSampler(value_space, distribution)

        if not Config.USE_GPU:
            # 샘플에서 positive 값이 제외되는지 확인
            for i in range(100):
                positive_idx = py.array([i % value_space.size])
                negative_samples = sampler.get_negative_samples(sample_size, positive_idx)
                self.assertNotIn(positive_idx, negative_samples)

                for sample in negative_samples[0]:
                    self.assertIn(sample, value_space)

        # 배치 처리 확인
        positive_idx_list = py.array([0, 1, 2, 3])
        negative_samples = sampler.get_negative_samples(sample_size, positive_idx_list)

        if not Config.USE_GPU:
            # 샘플에서 positive 값이 제외되는지 확인
            for i, positive_idx in enumerate(positive_idx_list):
                self.assertNotIn(positive_idx, negative_samples[i])
                for sample in negative_samples[i]:
                    self.assertIn(sample, value_space)

    def test_get_mixed_samples_and_labels(self):
        sample_size = 1000
        value_space = py.array([0, 1, 2, 3, 4])
        distribution = py.array([0.2, 0.2, 0.2, 0.2, 0.2])
        sampler = NegativeSampler(value_space, distribution)

        # 긍정적 예가 1개일 때
        positive_idx_list = py.array([1, 0, 1, 2, 0, 3])
        samples, labels = sampler.get_mixed_samples_and_labels(sample_size, positive_idx_list)

        correct_labels = py.empty_like(samples)
        for i, positive_idxes in enumerate(positive_idx_list):
            correct_labels[i] = (positive_idxes == samples[i])
        self.assertTrue(py.array_equal(correct_labels, labels))

        # 긍정적 예가 2개일 때
        positive_idx_list = py.array([[1, 0],
                                      [1, 2],
                                      [0, 3]])
        samples, labels = sampler.get_mixed_samples_and_labels(sample_size, positive_idx_list)

        correct_labels = py.empty_like(samples, dtype=int)
        for i, positive_idxes in enumerate(positive_idx_list):
            correct_labels[i] = py.logical_or((positive_idxes[0] == samples[i]), (positive_idxes[1] == samples[i]))
        self.assertTrue(py.array_equal(correct_labels, labels))
