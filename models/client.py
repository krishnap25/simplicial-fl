import warnings
import numpy as np


class Client:

    def __init__(self, client_id, group=None, train_data={'x': [], 'y': []}, eval_data={'x': [], 'y': []}, model=None, dataset='femnist'):
        self._model = model
        self.id = client_id  # integer
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        self.rng = model.rng  # use random number generator of the model

    def train(self, num_epochs=1, batch_size=10, minibatch=None, lr=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            averaged_loss: loss averaged over each stochastic update of the last epoch
            update: set of weights
        """
        if minibatch is None:
            data = self.train_data
            comp, update, averaged_loss = self.model.train(data, num_epochs, batch_size, lr)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac * len(self.train_data['x'])))
            # TODO: fix smapling from lists
            xs, ys = zip(*self.rng.sample(list(zip(self.train_data['x'], self.train_data['y'])), num_data))
            data = {'x': xs, 'y': ys}
            comp, update, averaged_loss = self.model.train(data, num_epochs, num_data, lr)
        num_train_samples = len(data['y'])
        return comp, num_train_samples, averaged_loss, update

    def test(self, model, train_and_test, split_by_user=True, train_users=True):
        """Tests model on self.eval_data.
        Args:
            model: model to measure metrics on
            train_and_test: If True, measure metrics on both train and test data. If False, only on test data.
        Return:
            dict of metrics returned by the model.
        """
        if split_by_user:
            if train_users:
                return model.test(self.train_data, split_by_user=split_by_user,
                              train_users=train_users)
            else:
                return model.test(self.eval_data, split_by_user=split_by_user,
                                  train_users=train_users)
        else:
            return model.test(self.eval_data, self.train_data if train_and_test else None, split_by_user=split_by_user,
                              train_users=train_users)

    def reinit_model(self):
        self._model.optimizer.initialize_w()

    @property
    def num_train_samples(self):
        return len(self.train_data['y'])

    @property
    def num_test_samples(self):
        return len(self.eval_data['y'])

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
