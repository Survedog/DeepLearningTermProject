from matplotlib import pyplot as plt
from config import Config
from common.utils import py
from math import ceil
import time

class Trainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = None

    def fit(self, x, t, batch_size=100, random_batch=True,
            max_epoch=20, print_info=True, eval_interval=1):

        iters_per_epoch = int(ceil(len(x) / batch_size))
        self.loss_list = []

        for epoch in range(1, max_epoch + 1):
            epoch_total_loss = 0.0
            epoch_loss_count = 0

            if random_batch:
                rand_idx = py.random.permutation(py.arange(len(x)))
                x = x[rand_idx]
                t = t[rand_idx]

            for iteration in range(1, iters_per_epoch + 1):
                start = time.time()
                x_batch, t_batch = self.get_batch(batch_size, iteration, x, t)

                loss = self.model.forward(x_batch, t_batch)
                epoch_total_loss += py.sum(loss)
                epoch_loss_count += len(x_batch)

                dout = py.ones(len(x_batch))
                self.model.backward(dout)
                self.optimizer.update(self.model.params, self.model.grads)

                if print_info:
                    print('[Trainer] Epoch[%d/%d] - Iteration[%d/%d] - 경과 시간: %.2fs' %
                          (epoch, max_epoch, iteration, iters_per_epoch, time.time() - start))

            if print_info and (epoch % eval_interval) == 0:
                self.eval_model(epoch, epoch_total_loss, epoch_loss_count)

        print('[Trainer] 최종 손실: %.3f' % (self.loss_list[-1]))

    def get_batch(self, batch_size, iteration, x, t):
        batch_start = (iteration - 1) * batch_size
        batch_end = min(iteration * batch_size, len(x))

        x_batch = x[batch_start:batch_end]
        t_batch = t[batch_start:batch_end]
        return x_batch, t_batch

    def eval_model(self, epoch, epoch_total_loss, loss_count):
        avg_loss = epoch_total_loss / loss_count
        print('[Trainer] Epoch %d - 평균 손실: %.3f' % (epoch, avg_loss))
        self.loss_list.append(avg_loss.get() if Config.USE_GPU else avg_loss)

    def plot(self, xlabel='epoch', ylabel='loss', title='Train Loss'):
        y = self.loss_list
        x = py.arange(len(y))

        if Config.USE_GPU:
            x = py.asnumpy(x)

        plt.plot(x, y, '.')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()


class RnnlmTrainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.perplexity_list = None  # eval: 모델의 평가치 (loss 등)
        self.time_idx = None

    def fit(self, xs, ts, time_size, batch_size=100, max_epoch=20, print_info=True, eval_interval=1):
        data_size = len(xs)
        iters_per_epoch = ceil(data_size / (time_size * batch_size))
        self.perplexity_list = []

        for epoch in range(1, max_epoch + 1):
            self.time_idx = 0
            epoch_total_loss = 0.0
            epoch_loss_count = 0
            self.model.reset_state()

            for iteration in range(1, iters_per_epoch + 1):
                start = time.time()
                xs_batch, ts_batch = self.get_batch(batch_size, time_size, xs, ts)

                loss = self.model.forward(xs_batch, ts_batch)
                epoch_total_loss += loss
                epoch_loss_count += 1
                
                # todo: 기울기 클리핑 적용
                self.model.backward()
                self.optimizer.update(self.model.params, self.model.grads)

                if print_info:
                    print('[Trainer] Epoch[%d/%d] - Iteration[%d/%d] - 경과 시간: %.2fs' %
                          (epoch, max_epoch, iteration, iters_per_epoch, time.time() - start))

            if print_info and (epoch % eval_interval) == 0:
                self.eval_model(epoch, epoch_total_loss, epoch_loss_count)

        print('[Trainer] 최종 퍼플렉서티: %.3f' % (self.perplexity_list[-1]))

    def get_batch(self, batch_size, time_size, xs, ts):
        data_size = len(xs)
        assert(time_size <= data_size)

        xs_batch = py.empty((batch_size, time_size), dtype=int)
        ts_batch = py.empty((batch_size, time_size), dtype=int)

        jump = data_size // batch_size
        offsets = [jump * i for i in range(batch_size)]

        for i, offset in enumerate(offsets):
            offset = (offset + self.time_idx) % data_size
            start = offset
            end = (offset + time_size) % data_size

            if (start < end):
                xs_batch[i] = xs[start:end]
                ts_batch[i] = ts[start:end]
            else:
                xs_batch[i] = py.hstack((xs[:end], xs[start:]))
                ts_batch[i] = py.hstack((ts[:end], ts[start:]))

        self.time_idx += time_size

        return xs_batch, ts_batch

    def eval_model(self, epoch, epoch_total_loss, loss_count):
        avg_loss = epoch_total_loss / loss_count
        perplexity = py.exp(avg_loss)
        print('[Trainer] Epoch %d - 평균 퍼플렉서티: %.3f' % (epoch, perplexity))
        self.perplexity_list.append(perplexity.get() if Config.USE_GPU else perplexity)

    def plot(self, xlabel='epoch', ylabel='perplexity', title='Train perplexity'):
        y = self.perplexity_list
        x = py.arange(len(y))

        if Config.USE_GPU:
            x = py.asnumpy(x)

        plt.plot(x, y, '.')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()
