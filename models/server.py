import numpy as np
import random

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY, AVG_LOSS_KEY
from baseline_constants import MAX_UPDATE_NORM


class Server:

    def __init__(self, model):
        self.model = model  # global model of the server.
        self.selected_clients = []
        self.updates = []
        self.rng = model.rng  # use random number generator of the model
        self.total_num_comm_rounds = 0
        self.eta = None

    def select_clients(self, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = self.rng.sample(possible_clients, num_clients)

        return [(len(c.train_data['y']), len(c.eval_data['y'])) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, lr=None, lmbda=None,
                    run_simplicial_fl=False, nonconformity_level=0.8, 
                    show_nb_selected_devices=None):

        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            lr: learning rate to use
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """

        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        losses = []

        if run_simplicial_fl:
            losses_before_training = self.eval_losses_on_train_clients(clients)
            weights_losses = self.clients_weights(clients=clients)
            thresh_loss = self.weighted_quantile(losses_before_training, weights_losses,
                                                 nonconformity_level=nonconformity_level)

            chosen_clients = [clients[i] for i in range(len(clients)) if losses_before_training[i] >= thresh_loss]
            print('\n\nChosen {} clients out of {}'.format(len(chosen_clients), len(clients)))
        else:
            chosen_clients = clients

        for c in chosen_clients:
            self.model.send_to([c])  # reset client model
            sys_metrics[c.id][BYTES_READ_KEY] += self.model.size
            if lmbda is not None:
                c._model.optimizer.lmbda = lmbda
            if lr is not None:
                c._model.optimizer.learning_rate = lr
            comp, num_samples, averaged_loss, update = c.train(num_epochs, batch_size, minibatch, lr)
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
            losses.append(averaged_loss)

            self.updates.append((num_samples, update))
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += self.model.size
            # sys_metrics[c.id][AVG_LOSS_KEY] = averaged_loss

        avg_loss = np.nan if len(losses) == 0 else \
            np.average(losses, weights=[len(c.train_data['y']) for c in chosen_clients])
        return sys_metrics, avg_loss, losses

    def update_model(self, aggregation, run_simplicial_fl=False, nonconformity_level=0.8, losses=None):

        if run_simplicial_fl:
            if len(self.updates) > 0:
                num_comm_rounds, is_updated = self.model.update(self.updates, aggregation,
                                                            max_update_norm=MAX_UPDATE_NORM,
                                                            maxiter=maxiter)
            else:
                num_comm_rounds = 0
                is_updated = False

        else:
            num_comm_rounds, is_updated = self.model.update(self.updates, aggregation,
                                                        max_update_norm=MAX_UPDATE_NORM)
        self.total_num_comm_rounds += num_comm_rounds
        self.updates = []
        return self.total_num_comm_rounds, is_updated

    def test_model(self, clients_to_test=None, train_and_test=False, split_by_user=True, train_users=True):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            train_and_test: If True, also measure metrics on training data
        """
        if clients_to_test is None:
            clients_to_test = self.selected_clients
        metrics = {}

        self.model.send_to(clients_to_test)

        for client in clients_to_test:
            c_metrics = client.test(self.model.cur_model, train_and_test, split_by_user=split_by_user, train_users=train_users)
            metrics[client.id] = c_metrics

        return metrics

    def get_clients_info(self, clients=None):
        """Returns the ids, hierarchies, num_train_samples and num_test_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients
        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_train_samples = {c.id: c.num_train_samples for c in clients}
        num_test_samples = {c.id: c.num_test_samples for c in clients}

        return ids, groups, num_train_samples, num_test_samples

    def eval_losses_on_train_clients(self, clients=None):
        # Implemented only when split_by_user is True
        losses = []

        if clients is None:
            clients = self.selected_clients

        self.model.send_to(clients)

        for c in clients:
            c_dict = c.test(self.model.cur_model, False, split_by_user=True, train_users=True)
            loss = c_dict['train_loss']
            losses.append(loss)

        return losses

    def clients_weights(self, clients=None):
        if clients is None:
            clients = self.selected_clients
        res = []
        for c in clients:
            res.append(len(c.train_data['y']))
        return res

    def weighted_quantile(self, losses, clients_weights, nonconformity_level=0.8):
        new_losses = []

        for i in range(len(losses)):
            for j in range(clients_weights[i]):
                new_losses.append(losses[i])

        return quantile(nonconformity_level, new_losses)


# helper function
def quantile(p, u):
    """ Computes the p-quantile of u
        :param ``float`` p: probability associated to the quantile
        :param ``numpy.array`` u: vector of realizations of the rqndom variable whose quantile is to be computed
        :return p-quantile of u
    """

    v = np.sort(u)

    if p == 0:
        return v[0]
    else:
        n = len(v)
        index = int(np.ceil(n*p)) - 1
        return v[index]
