import tensorflow as tf
import wandb

class LRLogger(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
      super(LRLogger, self).__init__()
      self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
      lr = self.optimizer.learning_rate(self.optimizer.iterations)
      wandb.log({"lr": lr.numpy}, commit=False)