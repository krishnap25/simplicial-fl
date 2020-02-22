from model import Model, Optimizer
import numpy as np



class ClientModel(Model):

    def __init__(self, lr, num_classes, max_batch_size=None, seed=None, optimizer=None):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(lr, seed, max_batch_size, optimizer=ErmOptimizer())


class ErmOptimizer(Optimizer):

    def __init__(self, starting_w=np.zeros(50)):
        super(ErmOptimizer, self).__init__(starting_w)
        self.optimizer_model = None
        self.learning_rate = 0.000001  # TODO: double check
        self.lmbda = 0.001

    def single_loss(self, x, y):
        pred = np.matmul(x.reshape(1, -1), self.w)
        loss = np.mean(y.reshape(1) * np.log(1 + np.exp(-pred)) + (1 - y.reshape(1)) * np.log(1 + np.exp(pred)))
        return loss

    def loss(self, batched_x, batched_y):
        n = len(batched_y)
        loss = 0.0
        for i in range(n):
            loss += self.single_loss(batched_x[i], batched_y[i])
        averaged_loss = loss / n
        return averaged_loss + 0.5 * self.lmbda/2 * np.linalg.norm(self.w)**2

    def gradient(self, x, y):  # x is only 1 image here

        image = x.reshape(1, -1)
        target = y.reshape(1)
        inp = np.matmul(image, self.w)
        loglossderiv = (-target / (1 + np.exp(inp)) + (1 - target) / (1 + np.exp(-inp))) / target.shape[0]
        return np.matmul(loglossderiv, image) + self.lmbda * self.w

    def run_step(self, batched_x, batched_y):
        loss = 0.0
        s = np.zeros(self.w.shape)
        n = len(batched_y)
        for i in range(n):
            s += self.learning_rate * self.gradient(batched_x[i], batched_y[i])
            loss += self.single_loss(batched_x[i], batched_y[i])
        self.w -= s/n
        averaged_loss = loss/n

        return averaged_loss

    def update_w(self):
        self.w_on_last_update = self.w

    def correct_single_label(self, x, y):
        proba = self.sigmoid(np.dot(x, self.w))
        if proba >= 0.5:
            prediction = 1.0
        else:
            prediction = 0.0

        return float(prediction == y)

    def initialize_w(self):
        self.w = np.zeros(50)
        self.w_on_last_update = np.zeros(50)

    def correct(self, x, y):
        nb_correct = 0.0
        for i in range(len(y)):
            nb_correct += self.correct_single_label(x[i], y[i])
        return nb_correct

    def size(self):
        return 50

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
