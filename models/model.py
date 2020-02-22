"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import random

from baseline_constants import ACCURACY_KEY, OptimLoggingKeys, AGGR_MEAN

from utils.model_utils import batch_data


class Model(ABC):

    def __init__(self, lr, seed, max_batch_size, optimizer=None):
        self.lr = lr
        self.optimizer = optimizer
        self.rng = random.Random(seed)
        self.size = None


        # largest batch size for which GPU will not run out of memory
        self.max_batch_size = max_batch_size if max_batch_size is not None else 2 ** 14
        print('***** using a max batch size of', self.max_batch_size)

        self.flops = 0

    def train(self, data, num_epochs=1, batch_size=10, lr=None):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
            averaged_loss: average of stochastic loss in the final epoch
        """
        if lr is None:
            lr = self.lr
        averaged_loss = 0.0

        batched_x, batched_y = batch_data(data, batch_size, rng=self.rng, shuffle=True)
        if self.optimizer.w is None:
            self.optimizer.initialize_w()

        for epoch in range(num_epochs):
            total_loss = 0.0

            for i, raw_x_batch in enumerate(batched_x):
                input_data = self.process_x(raw_x_batch)
                raw_y_batch = batched_y[i]
                target_data = self.process_y(raw_y_batch)

                loss = self.optimizer.run_step(input_data, target_data)
                total_loss += loss
            averaged_loss = total_loss / len(batched_x)
        # print('inner opt:', epoch, averaged_loss)

        self.optimizer.end_local_updates() # required for pytorch models
        update = np.copy(self.optimizer.w - self.optimizer.w_on_last_update)

        self.optimizer.update_w()

        comp = num_epochs * len(batched_y) * batch_size * self.flops
        return comp, update, averaged_loss

    def test(self, eval_data, train_data=None, split_by_user=True, train_users=True):
        """
        Tests the current model on the given data.
        Args:
            eval_data: dict of the form {'x': [list], 'y': [list]}
            train_data: None or same format as eval_data. If None, do not measure statistics on train_data.
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        if split_by_user:
            output = {'eval': [-float('inf'), -float('inf')], 'train': [-float('inf'), -float('inf')]}

            if self.optimizer.w is None:
                self.optimizer.initialize_w()

            total_loss, total_correct, count = 0.0, 0, 0
            batched_x, batched_y = batch_data(eval_data, self.max_batch_size, shuffle=False, eval_mode=True)
            for x, y in zip(batched_x, batched_y):
                x_vecs = self.process_x(x)
                labels = self.process_y(y)

                loss = self.optimizer.loss(x_vecs, labels)
                correct = self.optimizer.correct(x_vecs, labels)

                total_loss += loss * len(y)  # loss returns average over batch
                total_correct += correct  # eval_op returns sum over batch
                count += len(y)
                # counter_1 += 1
            loss = total_loss / count
            acc = total_correct / count
            if train_users:
                output['train'] = [loss, acc]
            else:
                output['eval'] = [loss, acc]

            return {
                    ACCURACY_KEY: output['eval'][1],
                    OptimLoggingKeys.TRAIN_LOSS_KEY: output['train'][0],
                    OptimLoggingKeys.TRAIN_ACCURACY_KEY: output['train'][1],
                    OptimLoggingKeys.EVAL_LOSS_KEY: output['eval'][0],
                    OptimLoggingKeys.EVAL_ACCURACY_KEY: output['eval'][1]
                    }
        else:
            data_lst = [eval_data] if train_data is None else [eval_data, train_data]
            output = {'eval': [-float('inf'), -float('inf')], 'train': [-float('inf'), -float('inf')]}

            if self.optimizer.w is None:
                self.optimizer.initialize_w()
            # counter_0 = 0
            for data, data_type in zip(data_lst, ['eval', 'train']):
                # counter_1 = 0
                total_loss, total_correct, count = 0.0, 0, 0
                batched_x, batched_y = batch_data(data, self.max_batch_size, shuffle=False, eval_mode=True)
                for x, y in zip(batched_x, batched_y):
                    x_vecs = self.process_x(x)
                    labels = self.process_y(y)

                    loss = self.optimizer.loss(x_vecs, labels)
                    correct = self.optimizer.correct(x_vecs, labels)

                    total_loss += loss * len(y)  # loss returns average over batch
                    total_correct += correct  # eval_op returns sum over batch
                    count += len(y)
                    # counter_1 += 1
                loss = total_loss / count
                acc = total_correct / count
                output[data_type] = [loss, acc]
                # counter_1 += 1

            return {ACCURACY_KEY: output['eval'][1],
                    OptimLoggingKeys.TRAIN_LOSS_KEY: output['train'][0],
                    OptimLoggingKeys.TRAIN_ACCURACY_KEY: output['train'][1],
                    OptimLoggingKeys.EVAL_LOSS_KEY: output['eval'][0],
                    OptimLoggingKeys.EVAL_ACCURACY_KEY: output['eval'][1]
                    }

    #def close(self):
    #    self.sess.close()

    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return np.asarray(raw_x_batch)

    # def process_y(self, raw_y_batch):
    #     """Pre-processes each batch of labels before being fed to the model."""
    #     res = []
    #     for i in range(len(raw_y_batch)):
    #         num = np.zeros(62) # Number of classes
    #         num[raw_y_batch[i]] = 1.0
    #         res.append(num)
    #     return np.asarray(res)
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return np.asarray(raw_y_batch)


class ServerModel:
    def __init__(self, model):
        self.model = model
        self.rng = model.rng

    @property
    def size(self):
        return self.model.optimizer.size()

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {}
        for c in clients:
            c.model.optimizer.reset_w(self.model.optimizer.w)
            c.model.size = self.model.optimizer.size()


    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = np.sum(weights)
        # Modif Here
        # weighted_updates = [np.zeros_like(v) for v in points[0]]

        weighted_updates = np.zeros_like(points[0])

        for w, p in zip(weights, points):
            weighted_updates += (w / tot_weights) * p

        return weighted_updates

    def update(self, updates, aggregation=AGGR_MEAN):
        """Updates server model using given client updates.

        Args:
            updates: list of (num_samples, update), where num_samples is the
                number of training samples corresponding to the update, and update
                is a list of variable weights
            aggregation: Algorithm used for aggregation. Allowed values are:
                [ 'mean'], i.e., only support aggregation with weighted mean
        """
        if len(updates) == 0:
            print('No updates obtained. Continuing without update')
            return 1, False
        def accept_update(u):
            # norm = np.linalg.norm([np.linalg.norm(x) for x in u[1]])
            norm = np.linalg.norm(u[1])
            return not (np.isinf(norm) or np.isnan(norm))
        all_updates = updates
        updates = [u for u in updates if accept_update(u)]
        if len(updates) < len(all_updates):
            print('Rejected {} individual updates because of NaN or Inf'.format(len(all_updates) - len(updates)))
        if len(updates) == 0:
            print('All individual updates rejected. Continuing without update')
            return 1, False

        points = [u[1] for u in updates]
        alphas = [u[0] for u in updates]
        if aggregation == AGGR_MEAN:
            weighted_updates = self.weighted_average_oracle(points, alphas)
            num_comm_rounds = 1
        else:
            raise ValueError('Unknown aggregation strategy: {}'.format(aggregation))

        # update_norm = np.linalg.norm([np.linalg.norm(v) for v in weighted_updates])
        update_norm = np.linalg.norm(weighted_updates)

        self.model.optimizer.w += np.array(weighted_updates)
        self.model.optimizer.reset_w(self.model.optimizer.w)  # update server model
        updated = True

        return num_comm_rounds, updated


class Optimizer(ABC):

    def __init__(self, starting_w=None, loss=None, loss_prime=None):
        self.w = starting_w
        self.w_on_last_update = np.copy(starting_w)
        self.optimizer_model = None

    @abstractmethod
    def loss(self, x, y):
        return None

    @abstractmethod
    def gradient(self, x, y):
        return None

    @abstractmethod
    def run_step(self, batched_x, batched_y): # should run a first order method step and return loss obtained
        return None

    @abstractmethod
    def correct(self, x, y):
        return None

    def end_local_updates(self):
        pass

    def reset_w(self, w):
        self. w = np.copy(w)
        self.w_on_last_update = np.copy(w)

