from model import Model, Optimizer
import numpy as np

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from utils.language_utils import bag_of_words, get_word_emb_arr, val_to_vec, split_line, line_to_indices
from utils.torch_utils import numpy_to_torch, torch_to_numpy

VOCAB_DIR = 'sent140/embs.json'


class ClientModel(Model):

    def __init__(self, lr, num_classes, max_batch_size=None, seed=None, optimizer=None):
        self.num_classes = num_classes
        self.seq_len = 20
        self.emb_array, self.word_indices, self.vocab = get_word_emb_arr(VOCAB_DIR)

        model = LSTMModel(self.emb_array)
        optimizer = ErmOptimizer(model)
        super(ClientModel, self).__init__(lr, seed, max_batch_size, optimizer=optimizer)

    def create_model(self):
        """Model function for linear model."""
        pass

    def process_x(self, raw_x_batch):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [line_to_indices(e, self.word_indices, self.seq_len) for e in x_batch]
        temp = np.asarray(x_batch)
        x_batch = torch.from_numpy(np.asarray(x_batch))
        return x_batch

    def process_y(self, raw_y_batch):
        #return torch.from_numpy(np.asarray(raw_y_batch, dtype=np.float32))
        return torch.from_numpy(np.asarray(raw_y_batch, dtype=np.float64))


class LSTMModel(torch.nn.Module):
    def __init__(self, word_emb_array, hidden_dim=-1,
                 n_lstm_layers=1, output_dim=1, default_batch_size=10):
        super(LSTMModel, self).__init__()

        torch.set_default_dtype(torch.float64)

        # Word embedding
        embedding_dim = word_emb_array.shape[1]
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.DoubleTensor(word_emb_array))

        # Hidden dimensions
        self.hidden_dim = hidden_dim if hidden_dim > 0 else embedding_dim

        # Number of stacked lstm layers
        self.n_lstm_layers = n_lstm_layers

        # shape of input/output tensors: (batch_dim, seq_dim, feature_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, self.hidden_dim, n_lstm_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim, output_dim)

        # hidden state and cell state
        self.h0 = torch.zeros(self.n_lstm_layers, default_batch_size, self.hidden_dim).requires_grad_()
        self.c0 = torch.zeros(self.n_lstm_layers, default_batch_size, self.hidden_dim).requires_grad_()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        # word embedding
        x = self.embedding(x)

        if self.h0.size(1) == x.size(0):
            self.h0.data.zero_()
            self.c0.data.zero_()
        else:
            # resize hidden vars
            self.h0 = torch.zeros(self.n_lstm_layers, x.size(0), self.hidden_dim).requires_grad_()
            self.c0 = torch.zeros(self.n_lstm_layers, x.size(0), self.hidden_dim).requires_grad_()

        # lstm
        out, _ = self.lstm(x, (self.h0.detach(), self.c0.detach()))

        # Index hidden state of last time step; out.size = `batch, seq_len, hidden`
        out = self.fc(out[:, -1, :])
        return out.view(-1)  # hard-coded for binary classification


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

    def _l2_reg_penalty(self):
        loss = sum([torch.norm(p)**2  for p in self.model.trainable_parameters()])
        return 0.5 * self.lmbda * loss

    def loss(self, x, y):
        """Compute batch loss on proceesed batch (x, y)"""
        with torch.no_grad():
            preds = self.model(x)
            loss = binary_cross_entropy_with_logits(preds, y) + self._l2_reg_penalty()
        return loss.item()

    def gradient(self, x, y):
        preds = self.model(x)
        loss = binary_cross_entropy_with_logits(preds, y) + self._l2_reg_penalty()
        gradient = torch.autograd.grad(loss, self.model.trainable_parameters())
        return gradient

    def loss_and_gradient(self, x, y):
        preds = self.model(x)
        loss = binary_cross_entropy_with_logits(preds, y) + self._l2_reg_penalty()
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
            preds = self.model(x)
            predicted_labels = (torch.sign(preds) + 1) / 2
            return (predicted_labels == y).sum().item()

    def size(self):
        return len(self.w)
