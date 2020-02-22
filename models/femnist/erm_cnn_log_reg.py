from collections import OrderedDict
import copy
import numpy as np
from model import Model, Optimizer

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from utils.torch_utils import numpy_to_torch, torch_to_numpy


class ClientModel(Model):

    def __init__(self, lr, num_classes,
                 max_batch_size=None, seed=None, optimizer=None):
        self.num_classes = num_classes
        self.device = torch.device('cpu') # TODO: set device throughout
        model = ConvNetModel().to(self.device)
        optimizer = ErmOptimizer(model)
        super(ClientModel, self).__init__(lr, seed, max_batch_size,
                                          optimizer=optimizer)

    def set_device(self, device):
        self.device = device
        self.optimizer.set_device(device)

    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return torch.from_numpy(
            np.asarray(raw_x_batch, dtype=np.float32).reshape(-1, 1, 28, 28)
        ).to(self.device)

    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        # return torch.from_numpy(np.asarray(raw_y_batch, dtype=np.float32), device=device)
        return torch.LongTensor(raw_y_batch).to(self.device)


class ConvNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 32, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(32, 64, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        ]))

        self.fc = nn.Linear(1024, 62)
        ## old:
        #self.fc = nn.Sequential(OrderedDict([
        #    ('f5', nn.Linear(1024, 256)),
        #    ('relu6', nn.ReLU()),
        #    ('f7', nn.Linear(256, 62)),
        #]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]



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
        self.w = np.copy(w)
        self.w_on_last_update = np.copy(w)
        numpy_to_torch(self.w, self.model)

    def end_local_updates(self):
        """self.model is updated by iterations; update self.w to make it consistent with this"""
        self.w = torch_to_numpy(self.model.trainable_parameters())

    def update_w(self):
        self.w_on_last_update = self.w

    def _l2_reg_penalty(self):
        # TODO: note: no l2penalty is applied to the convnet
        # loss = sum([torch.norm(p)**2  for p in self.model.trainable_parameters()])
        # return 0.5 * self.lmbda * loss
        return 0.0

    def loss(self, x, y):
        """Compute batch loss on proceesed batch (x, y)"""
        with torch.no_grad():
            preds = self.model(x)
            loss = cross_entropy(preds, y) + self._l2_reg_penalty()
        return loss.item()

    def gradient(self, x, y):
        preds = self.model(x)
        loss = cross_entropy(preds, y) + self._l2_reg_penalty()
        gradient = torch.autograd.grad(loss, self.model.trainable_parameters())
        return gradient

    def loss_and_gradient(self, x, y):
        preds = self.model(x)
        loss = cross_entropy(preds, y) + self._l2_reg_penalty()
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
