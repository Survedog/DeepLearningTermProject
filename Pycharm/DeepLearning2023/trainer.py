from utils import np
from config import Config


class Trainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    # todo: 랜덤 미니배치 처리 구현
    def fit(self, train_data, answer, batch_size=Config.DEFAULT_BATCH_SIZE,
            max_epoch=Config.DEFAULT_MAX_EPOCH, print_info=True, print_interval=1):

        iters_per_epoch = 1 + train_data.shape[0] // batch_size
        final_loss = 0.0

        for epoch in range(max_epoch):
            epoch_total_loss = 0.0
            loss_count = 0

            for iteration in range(iters_per_epoch):
                batch_start = iteration * batch_size
                batch_end = min((iteration + 1) * batch_size, train_data.shape[0])

                data_batch = train_data[batch_start:batch_end]
                answer_batch = answer[batch_start:batch_end]
                loss = self.model.forward(data_batch, answer_batch)

                epoch_total_loss += np.sum(loss) / batch_size
                loss_count += 1

                dout = np.ones(batch_end - batch_start)
                self.model.backward(dout)
                self.optimizer.update(self.model.params, self.model.grads)

            if print_info and (epoch % print_interval) == 0:
                avg_loss = epoch_total_loss / loss_count
                print('[Epoch %d] 평균 손실: %.2f' % (epoch, avg_loss))
                final_loss = avg_loss

        print('[RESULT] 최종 손실: %.2f' % (final_loss))
