
class HyperParams:
    '''
    Model training hyper parameters.

    Parameters
    ----------
    epoch_count: int
    learning_rate: int
    batch_size : int
    '''
    def __init__(self, epoch_count, learning_rate, batch_size):
        self.epoch_count = epoch_count
        self.learning_rate = learning_rate
        self.batch_size = batch_size
