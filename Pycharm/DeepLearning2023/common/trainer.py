from matplotlib import pyplot as plt
from config import Config
from common.utils import py
import time

class Trainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = None

    def fit(self, x, t, batch_size=100, random_batch=True,
            max_epoch=20, print_info=True, eval_interval=1):

        iters_per_epoch = int(py.ceil(len(x) / batch_size))
        self.loss_list = []

        for epoch in range(1, max_epoch + 1):
            epoch_total_loss = 0.0
            loss_count = 0

            if random_batch:
                rand_idx = py.random.permutation(py.arange(len(x)))
                x = x[rand_idx]
                t = t[rand_idx]

            for iteration in range(1, iters_per_epoch + 1):
                start = time.time()
                x_batch, t_batch = self.get_batch(batch_size, iteration, x, t)

                loss = self.model.forward(x_batch, t_batch)
                epoch_total_loss += py.sum(loss)
                loss_count += batch_size

                dout = py.ones(len(x_batch))
                self.model.backward(dout)
                self.optimizer.update(self.model.params, self.model.grads)

                if print_info:
                    print('[Trainer] Epoch[%d/%d] - Iteration[%d/%d] - 경과 시간: %.2fs' %
                          (epoch, max_epoch, iteration, iters_per_epoch, time.time() - start))

            if print_info and (epoch % eval_interval) == 0:
                self.eval_model(epoch, epoch_total_loss, loss_count)

        print('[Trainer] 최종 손실: %.3f' % (self.loss_list[-1]))

    def get_batch(self, batch_size, iteration, x, t):
        batch_start = (iteration - 1) * batch_size
        batch_end = min(iteration * batch_size, len(x))

        x_batch = x[batch_start:batch_end]
        t_batch = t[batch_start:batch_end]
        loss = self.model.forward(x_batch, t_batch)
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


class RnnTrainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.pp_list = None  # eval: 모델의 평가치 (loss 등)

    def fit(self, x, t, batch_size=100, max_epoch=20, print_info=True, eval_interval=1):

        iters_per_epoch = int(py.ceil(len(x) / batch_size))
        self.pp_list = []

        for epoch in range(1, max_epoch + 1):
            epoch_total_loss = 0.0
            loss_count = 0

            for iteration in range(1, iters_per_epoch + 1):
                start = time.time()
                data_batch, answer_batch = self.get_batch(batch_size, iteration, x, t)

                loss = self.model.forward(data_batch, answer_batch)
                epoch_total_loss += py.sum(loss)
                loss_count += batch_size

                dout = py.ones(len(data_batch))
                self.model.backward(dout)
                self.optimizer.update(self.model.params, self.model.grads)

                if print_info:
                    print('[Trainer] Epoch[%d/%d] - Iteration[%d/%d] - 경과 시간: %.2fs' %
                          (epoch, max_epoch, iteration, iters_per_epoch, time.time() - start))

            if print_info and (epoch % eval_interval) == 0:
                self.eval_model(epoch, epoch_total_loss, loss_count)

        self.print_final_eval()

    def get_batch(self, batch_size, iteration, x, t):
        x =
        x_batch = x[batch_start:batch_end]
        t_batch = t[batch_start:batch_end]
        loss = self.model.forward(x_batch, t_batch)
        return x_batch, t_batch

    def eval_model(self, epoch, epoch_total_loss, loss_count):
        avg_loss = epoch_total_loss / loss_count
        print('[Trainer] Epoch %d - 평균 퍼플렉서티: %.3f' % (epoch, avg_loss))
        self.pp_list.append(avg_loss.get() if Config.USE_GPU else avg_loss)

    def print_final_eval(self):
        print('[Trainer] 최종 퍼플렉서티: %.3f' % (self.pp_list[-1]))