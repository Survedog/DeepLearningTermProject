import numpy
from utils import py
from base_optimizer import OptimizerBase



class AdamOptimizer(OptimizerBase):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        super().__init__(learning_rate)
        self.m = []
        self.v = []
        self.beta1 = beta1
        self.beta2 = beta2
        self.timestep = 0

    def update(self, params, grads, epsilon=1e-8):
        if not self.m:
            for param in params:
                self.m.append(py.zeros_like(param))
                self.v.append(py.zeros_like(param))

        beta1, beta2 = self.beta1, self.beta2
        self.timestep += 1

        for i in range(len(params)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grads[i]
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * py.square(grads[i])

            unbiased_m = self.m[i] / (1 - beta1 ** self.timestep)
            unbiased_v = self.v[i] / (1 - beta2 ** self.timestep)

            params[i] -= self.learning_rate * unbiased_m / (py.sqrt(unbiased_v) + epsilon)
