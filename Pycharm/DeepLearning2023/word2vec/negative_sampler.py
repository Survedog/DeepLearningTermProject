from config import Config
from common.utils import py
import numpy
import time


class NegativeSampler:

    def __init__(self, value_space, distribution, power=0.75):
        self.value_space = value_space
        self.distribution = py.power(distribution, power)
        self.distribution /= py.sum(self.distribution)

        # 랜덤 Generator를 통한 추출을 위해 numpy를 사용.
        if Config.USE_GPU:
            self.distribution = py.asnumpy(self.distribution)
            self.value_space = py.asnumpy(value_space.get())

        self.rand_generator = None

    def get_negative_samples(self, sample_size, positive_idx_list):
        """
        :param sample_size: 배치 별 추출할 샘플 개수
        :param positive_idx_list: 각 샘플들에 포함되지 않는 값의 value space 상 인덱스
        :return: positive 값이 포함되지 않은 랜덤 추출 샘플 / 각 샘플의 정답 label(0/1)
        """
        if self.rand_generator is None:
            self.rand_generator = numpy.random.default_rng(int(time.time()))

        negative_samples = py.empty([positive_idx_list.shape[0], sample_size], dtype=self.value_space.dtype)

        if Config.USE_GPU:  # cupy 사용 시 속도를 위해 긍정적 샘플 중복 허용
            negative_samples = py.asarray(self.rand_generator.choice(self.value_space,
                                                                     (positive_idx_list.shape[0], sample_size),
                                                                     replace=True,
                                                                     p=self.distribution))
        else:
            for i, positive_idx in enumerate(positive_idx_list):
                p_positive = self.distribution[positive_idx]
                self.distribution[positive_idx] = 0.0

                negative_samples[i] = self.rand_generator.choice(self.value_space,
                                                                 sample_size,
                                                                 replace=True,
                                                                 p=self.distribution / (1 - p_positive))

                self.distribution[positive_idx] = p_positive

        return negative_samples

    def get_mixed_samples_and_labels(self, sample_size, positive_idx_list):
        if positive_idx_list.ndim == 1:
            positive_idx_list = positive_idx_list.reshape(-1, 1)

        negative_samples = self.get_negative_samples(sample_size - positive_idx_list.shape[-1], positive_idx_list)
        mixed_samples = py.hstack((positive_idx_list, negative_samples))

        labels = py.zeros((positive_idx_list.shape[0], sample_size), dtype=int)
        if Config.USE_GPU:  # 부정적/긍정적 샘플이 섞여 있음
            for i in range(positive_idx_list.shape[-1]):
                positive_mask = py.repeat(positive_idx_list[:, i][py.newaxis].T, sample_size, axis=-1)
                labels = py.logical_or(labels, mixed_samples == positive_mask)
            labels = labels.astype(int)
        else:  # 부정적/긍정적 샘플이 분리되어 있음
            labels[:, py.arange(positive_idx_list.shape[-1])] = 1

        return mixed_samples, labels
