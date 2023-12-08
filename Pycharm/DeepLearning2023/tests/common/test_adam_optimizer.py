import unittest
from common.adam_optimizer import *


class AdamOptimizerTests(unittest.TestCase):

    def test_update(self):
        beta1 = 0.9
        beta2 = 0.999
        learning_rate = 0.001

        optimizer = AdamOptimizer(learning_rate, beta1, beta2)

        w1 = py.array([[1.0, 0.5, 0.3],
                       [0.7, 0.4, 0.8]])
        w2 = py.array([[1.3, 0.5],
                       [2.4, 1.8],
                       [0.1, 3.5]])
        g1 = py.array([[1.0, -1.0, 1.0],
                       [-0.5, 0.5, -0.5]])
        g2 = py.array([[1.0, 0.0],
                       [0.0, -2.0],
                       [0.0, 0.0]])

        params = [w1, w2]
        grads = [g1, g2]

        w1_old = py.array(w1)
        w2_old = py.array(w2)
        optimizer.update(params, grads)

        # 첫 시행 시 기울기가 지정된 매개변수들만 갱신됨.
        self.assertTrue(py.array_equal(w1 < w1_old, g1 > 0.0))
        self.assertTrue(py.array_equal(w1 > w1_old, g1 < 0.0))
        self.assertTrue(py.array_equal(w2 < w2_old, g2 > 0.0))
        self.assertTrue(py.array_equal(w2 > w2_old, g2 < 0.0))

        g1[...] = 0
        g2[...] = 0
        w1_old[...] = w1
        w2_old[...] = w2
        optimizer.update(params, grads)

        # 두번째 실행 시 기울기가 0이어도 가속도에 의해 갱신됨.
        self.assertFalse(py.array_equal(w1, w1_old))
        self.assertFalse(py.array_equal(w2, w2_old))
