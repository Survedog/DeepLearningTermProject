from utils import np


class NegativeSampler:

    def __init__(self, value_space, distribution):
        self.value_space = value_space
        self.distribution = distribution

    def get_negative_samples(self, sample_size, positive_idx_list):
        """
        :param sample_size: 배치 별 추출할 샘플 개수
        :param positive_idx_list: 각 샘플에 포함되지 않는 값의 value space 인덱스들
        :return: positive 값이 포함되지 않은 랜덤 추출 샘플 / 각 샘플의 정답 label(0/1)
        """
        negative_samples = np.empty([positive_idx_list.shape[0], sample_size])

        for i, positive_idx in enumerate(positive_idx_list):
            p_positive = self.distribution[positive_idx]
            self.distribution[positive_idx] = 0.0

            negative_samples[i] = np.random.choice(self.value_space,
                                                   sample_size,
                                                   replace=True,
                                                   p=self.distribution / (1 - p_positive))

            self.distribution[positive_idx] = p_positive

        bool_label = np.zeros_like(negative_samples)
        bool_label[:, 0] = 1
        return negative_samples, bool_label
