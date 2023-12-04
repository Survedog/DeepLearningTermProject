from utils import np


class NegativeSampler:

    def __init__(self, value_space, distribution):
        self.value_space = value_space
        self.distribution = distribution

    def get_negative_samples(self, sample_size, positive_idx_list):
        """
        :param sample_size: 배치 별 추출할 샘플 개수
        :param positive_idx_list: 각 샘플들에 포함되지 않는 값의 value space 상 인덱스
        :return: positive 값이 포함되지 않은 랜덤 추출 샘플 / 각 샘플의 정답 label(0/1)
        """
        negative_samples = np.empty([positive_idx_list.shape[0], sample_size], dtype=self.value_space.dtype)

        for i, positive_idx in enumerate(positive_idx_list):
            p_positive = self.distribution[positive_idx]
            self.distribution[positive_idx] = 0.0

            negative_samples[i] = np.random.choice(self.value_space,
                                                   sample_size,
                                                   replace=True,
                                                   p=self.distribution / (1 - p_positive))

            self.distribution[positive_idx] = p_positive

        return negative_samples

    def get_mixed_samples_and_labels(self, sample_size, positive_idx_list):
        negative_samples = self.get_negative_samples(sample_size - 1, positive_idx_list)
        mixed_samples = np.hstack((positive_idx_list[np.newaxis].T, negative_samples))

        labels = np.zeros_like(mixed_samples)
        labels[:, 0] = 1

        return mixed_samples, labels
