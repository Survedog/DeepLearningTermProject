import unittest
from adam_optimizer import *


class AdamOptimizerTests(unittest.TestCase):

    def test_update(self):
        optimizer = AdamOptimizer()

        w1 = np.array([[1.0, 0.5, 0.3],
                       [0.7, 0.4, 0.8]])
        w2 = np.array([[1.3, 0.5],
                       [2.4, 1.8],
                       [0.1, 3.5]])
        params = [w1, w2]

        g1 = np.array([[1.0, 1.0, 1.0],
                       [0.5, 0.5, 0.5]])
        g2 = np.array([[1.0, 0.0],
                       [0.0, 2.0],
                       [0.0, 0.0]])
        grads = [g1, g2]

        optimizer.update(params, grads)
        

