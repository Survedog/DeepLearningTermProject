from matplotlib import pyplot as plt
from config import Config
from common.utils import py
import time

class Trainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.eval_list = None  # eval: 모델의 평가치 (loss 등)

    def fit(self, x, t, batch_size=100, random_batch=True,
            max_epoch=20, print_info=True, eval_interval=1):

        iters_per_epoch = int(py.ceil(x.shape[0] / batch_size))
        self.eval_list = []

        for epoch in range(1, max_epoch + 1):
            epoch_total_loss = 0.0
            loss_count = 0

            if random_batch:
                rand_idx = py.random.permutation(py.arange(len(x)))
                x = x[rand_idx]
                t = t[rand_idx]

            for iteration in range(1, iters_per_epoch + 1):
                start = time.time()
                data_batch, answer_batch = self.get_batch(batch_size, iteration, x, t)

                loss = self.model.forward(data_batch, answer_batch)
                epoch_total_loss += py.sum(loss)
                loss_count += batch_size

                dout = py.ones(data_batch.shape[0])
                self.model.backward(dout)
                self.optimizer.update(self.model.params, self.model.grads)

                if print_info:
                    print('[Trainer] Epoch[%d/%d] - Iteration[%d/%d] - 경과 시간: %.2fs' %
                          (epoch, max_epoch, iteration, iters_per_epoch, time.time() - start))

            if print_info and (epoch % eval_interval) == 0:
                self.eval_model(epoch, epoch_total_loss, loss_count)

        self.print_final_eval()

    def get_batch(self, batch_size, iteration, train_data, answer):
        batch_start = (iteration - 1) * batch_size
        batch_end = min(iteration * batch_size, train_data.shape[0])

        data_batch = train_data[batch_start:batch_end]
        answer_batch = answer[batch_start:batch_end]
        loss = self.model.forward(data_batch, answer_batch)
        return data_batch, answer_batch

    def eval_model(self, epoch, epoch_total_loss, loss_count):
        avg_loss = epoch_total_loss / loss_count
        print('[Trainer] Epoch %d - 평균 손실: %.3f' % (epoch, avg_loss))
        self.eval_list.append(avg_loss.get() if Config.USE_GPU else avg_loss)

    def print_final_eval(self):
        print('[Trainer] 최종 손실: %.3f' % (self.eval_list[-1]))

    def plot(self, xlabel='epoch', ylabel='loss', title='Train Loss'):
        y = self.eval_list
        x = py.arange(len(y))

        plt.plot(x, y, '.')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()


class RnnTrainer(Trainer):

    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def get_batch(self, batch_size, iteration, train_data, answer):
        data_batch = train_data[batch_start:batch_end]
        answer_batch = answer[batch_start:batch_end]
        loss = self.model.forward(data_batch, answer_batch)
        return data_batch, answer_batch

    def eval_model(self, epoch, epoch_total_loss, loss_count):
        avg_loss = epoch_total_loss / loss_count
        print('[Trainer] Epoch %d - 평균 퍼플렉서티: %.3f' % (epoch, avg_loss))
        self.eval_list.append(avg_loss.get() if Config.USE_GPU else avg_loss)

    def print_final_eval(self):
        print('[Trainer] 최종 퍼플렉서티: %.3f' % (self.eval_list[-1]))