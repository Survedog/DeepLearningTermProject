from utils import np
from config import Config


class Trainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def fit(self, train_data, answer, batch_size=Config.DEFAULT_BATCH_SIZE,
            max_epoch=Config.DEFAULT_MAX_EPOCH, print_info=True, print_interval=32):

        iters_per_epoch = 1 + train_data.shape[0] // batch_size

        for epoch in range(max_epoch):
            epoch_total_loss = 0.0
            loss_count = 0

            for iteration in range(iters_per_epoch):
                batch_start = iteration * batch_size
                batch_end = min((iteration + 1) * batch_size, train_data.shape[0])
                print('batch start: %d, end: %d' % (batch_start, batch_end))

                data_batch = train_data[batch_start:batch_end]
                answer_batch = answer[batch_start:batch_end]
                loss = self.model.forward(data_batch, answer_batch)
                self.model.backward(1.0)

                self.optimizer.update(self.model.params, self.model.grads)
                epoch_total_loss += loss
                loss_count += 1

            if print_info and (epoch % print_interval) == 0:
                avg_loss = epoch_total_loss / loss_count
                print('[Epoch %d] 평균 손실: %.2f' % (epoch, avg_loss))
