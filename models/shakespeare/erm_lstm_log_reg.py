"""
Recurrent model for character-level language modeling for the Shakespeare dataset
"""
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from model import Model, Optimizer
from utils.language_utils import letter_to_vec, word_to_indices
from utils.torch_utils import numpy_to_torch, torch_to_numpy

class ClientModel(Model):
    def __init__(self, lr, seq_len, num_classes, n_hidden,
                 n_recurrent_layers=2, max_batch_size=None, seed=None):

        model = RecurrentModel(num_classes, n_hidden, n_recurrent_layers, n_hidden)
        self.device = torch.device('cpu')
        model = model.to(self.device)
        optimizer = ErmOptimizer(model)
        # TODO: set device throughout
        super(ClientModel, self).__init__(lr, seed, max_batch_size, optimizer=optimizer)

    def set_device(self, device):
        self.device = device
        self.optimizer.set_device(device)

    def process_x(self, raw_x_batch):
        return torch.LongTensor(raw_x_batch).to(self.device)

    def process_y(self, raw_y_batch):
        return torch.LongTensor(raw_y_batch).to(self.device)

class RecurrentModel(torch.nn.Module):
    def __init__(self, num_classes, hidden_dim=128,
                 n_recurrent_layers=1, output_dim=128, default_batch_size=32):
        super(RecurrentModel, self).__init__()

        # Word embedding
        embedding_dim = 8
        self.embedding = torch.nn.Embedding(num_classes, embedding_dim)

        # Hidden dimensions
        self.hidden_dim = hidden_dim if hidden_dim > 0 else embedding_dim

        # Number of stacked lstm layers
        self.n_recurrent_layers = n_recurrent_layers

        # shape of input/output tensors: (batch_dim, seq_dim, feature_dim)
        self.rnn = torch.nn.GRU(embedding_dim, self.hidden_dim, n_recurrent_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim, output_dim)

        # hidden state and cell state (cell state is in LSTM only)
        self.h0 = torch.zeros(self.n_recurrent_layers, default_batch_size, self.hidden_dim).requires_grad_()
        # self.c0 = torch.zeros(self.n_recurrent_layers, default_batch_size, self.hidden_dim).requires_grad_()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        # word embedding
        x = self.embedding(x)

        if self.h0.size(1) == x.size(0):
            self.h0.data.zero_()
            # self.c0.data.zero_()
        else:
            # resize hidden vars
            device = next(self.parameters()).device
            self.h0 = torch.zeros(self.n_recurrent_layers, x.size(0),
                                  self.hidden_dim).to(device).requires_grad_()
            # self.c0 = torch.zeros(self.n_recurrent_layers, x.size(0),
            #                       self.hidden_dim).to(device).requires_grad_()

        # query RNN
        out, _ = self.rnn(x, self.h0.detach())
        # out, _ = self.rnn(x, (self.h0.detach(), self.c0.detach()))

        # Index hidden state of last time step; out.size = `batch, seq_len, hidden`
        out = self.fc(out[:, -1, :])
        return out

    def to(self, device):
        super().to(device)
        self.h0 = self.h0.to(device)
        return self
        # self.c0 = self.c0.to(device)

class ErmOptimizer(Optimizer):

    def __init__(self, model):
        super(ErmOptimizer, self).__init__(torch_to_numpy(model.trainable_parameters()))
        self.optimizer_model = None
        self.learning_rate = None
        self.lmbda = None
        self.model = model

    def initialize_w(self):
        self.w = torch_to_numpy(self.model.trainable_parameters())
        self.w_on_last_update = np.copy(self.w)

    def reset_w(self, w):
        """w is provided by server; update self.model to make it consistent with this"""
        self. w = np.copy(w)
        self.w_on_last_update = np.copy(w)
        numpy_to_torch(self.w, self.model)

    def end_local_updates(self):
        """self.model is updated by iterations; update self.w to make it consistent with this"""
        self.w = torch_to_numpy(self.model.trainable_parameters())

    def update_w(self):
        self.w_on_last_update = self.w

    def loss(self, x, y):
        """Compute batch loss on proceesed batch (x, y)"""
        with torch.no_grad():
            preds = self.model(x)
            loss = cross_entropy(preds, y)
        return loss.item()

    def gradient(self, x, y):
        preds = self.model(x)
        loss = cross_entropy(preds, y)
        gradient = torch.autograd.grad(loss, self.model.trainable_parameters())
        return gradient

    def loss_and_gradient(self, x, y):
        preds = self.model(x)
        loss = cross_entropy(preds, y)
        gradient = torch.autograd.grad(loss, self.model.trainable_parameters())
        return loss, gradient

    def run_step(self, batched_x, batched_y):
        """Run single gradient step on (batched_x, batched_y) and return loss encountered"""
        loss, gradient = self.loss_and_gradient(batched_x, batched_y)
        for p, g in zip(self.model.trainable_parameters(), gradient):
            p.data -= self.learning_rate * g.data

        return loss.item()

    def correct(self, x, y):
        with torch.no_grad():
            outputs = self.model(x)
            pred = outputs.argmax(dim=1, keepdim=True)
            return pred.eq(y.view_as(pred)).sum().item()

    def size(self):
        return len(self.w)

    def set_device(self, device):
        self.model = self.model.to(device)
