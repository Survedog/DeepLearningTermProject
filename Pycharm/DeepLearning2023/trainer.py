from matplotlib import pyplot as plt
from config import Config
from utils import py
import numpy
import time

class Trainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = None

    def fit(self, train_data, answer, batch_size=Config.DEFAULT_BATCH_SIZE, random_batch=True,
            max_epoch=Config.DEFAULT_MAX_EPOCH, print_info=True, print_interval=1):

        iters_per_epoch = int(py.ceil(train_data.shape[0] / batch_size))
        self.loss_list = []

        for epoch in range(1, max_epoch + 1):
            epoch_total_loss = 0.0
            loss_count = 0

            if random_batch:
                rand_idx = py.random.permutation(py.arange(len(train_data)))
                train_data = train_data[rand_idx]
                answer = answer[rand_idx]

            for iteration in range(1, iters_per_epoch + 1):
                start = time.time()
                batch_start = (iteration - 1) * batch_size
                batch_end = min(iteration * batch_size, train_data.shape[0])

                data_batch = train_data[batch_start:batch_end]
                answer_batch = answer[batch_start:batch_end]
                loss = self.model.forward(data_batch, answer_batch)

                epoch_total_loss += py.sum(loss) / batch_size
                loss_count += 1

                dout = py.ones(batch_end - batch_start)
                self.model.backward(dout)
                self.optimizer.update(self.model.params, self.model.grads)

                if print_info:
                    print('[Trainer] Epoch %d - Iteration[%d/%d] - 경과 시간: %.2fs' % (epoch, iteration, iters_per_epoch, time.time() - start))

            if print_info and (epoch % print_interval) == 0:
                avg_loss = epoch_total_loss / loss_count
                print('[Trainer] Epoch %d - 평균 손실: %.3f' % (epoch, avg_loss))

                self.loss_list.append(avg_loss.get() if Config.USE_GPU else avg_loss)

        print('[Trainer] 최종 손실: %.3f' % (self.loss_list[-1]))

    def plot_loss(self):
        y = self.loss_list
        x = numpy.arange(len(y))

        plt.plot(x, y, '.')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('CBow Train Loss')
        plt.show()

