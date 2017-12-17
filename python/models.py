import numpy as np
from numpy.matlib import repmat
import mxnet as mx
from mxnet import nd, autograd
import os
import random
from widgets import ProgressBar

class DQNNetwork:
  def __init__(self, sizes=[], batch_size=32, alpha=0.01, use_gpu=False,
      progbar=True, momentum=0.0):
    self.batch_size = batch_size
    self.sizes = sizes
    self.alpha = alpha
    self.moving_loss = 0.0
    self.use_gpu = use_gpu
    self.progbar = progbar
    self.ctx = None
    if use_gpu:
      self.ctx = mx.gpu(0)
    else:
      self.ctx = mx.cpu()

    # define weights
    s = sizes
    self.Wdense1 = nd.random_normal(shape=(s[0], s[1]), ctx=self.ctx) * 0.01
    self.bdense1 = nd.random_normal(shape=(s[1]), ctx=self.ctx) * 0.01

    self.Wdense2 = nd.random_normal(shape=(s[1], s[2]), ctx=self.ctx) * 0.01
    self.bdense2 = nd.random_normal(shape=(s[2]), ctx=self.ctx) * 0.01

    self.Wdense3 = nd.random_normal(shape=(s[2], s[3]), ctx=self.ctx) * 0.01
    self.bdense3 = nd.random_normal(shape=(s[3]), ctx=self.ctx) * 0.01

    self.Wdense4 = nd.random_normal(shape=(s[3], s[4]), ctx=self.ctx) * 0.01
    self.bdense4 = nd.random_normal(shape=(s[4]), ctx=self.ctx) * 0.01

    self.Wdense5 = nd.random_normal(shape=(s[4], s[5]), ctx=self.ctx) * 0.01
    self.bdense5 = nd.random_normal(shape=(s[5]), ctx=self.ctx) * 0.01

    self.params = [
        self.Wdense1, self.bdense1,
        self.Wdense2, self.bdense2,
        self.Wdense3, self.bdense3,
        self.Wdense4, self.bdense4,
        self.Wdense5, self.bdense5
        ]
    for param in self.params:
      param.attach_grad()
    self.moments = [nd.zeros(param.shape, ctx=self.ctx)
        for param in self.params]
    self.momentum = momentum

    # define architecture
    def fc_architecture(inputs, state={}):
      dense1 = nd.relu(nd.dot(inputs, self.Wdense1) + self.bdense1)
      dense2 = nd.relu(nd.dot(dense1, self.Wdense2) + self.bdense2)
      dense3 = nd.relu(nd.dot(dense2, self.Wdense3) + self.bdense3)
      dense4 = nd.relu(nd.dot(dense3, self.Wdense4) + self.bdense4)
      qvalues = nd.dot(dense4, self.Wdense5) + self.bdense5
      return (qvalues, {})

    # define loss
    def sqerr(p, q): # p is the real distribution, q is the predicted
      return nd.mean(nd.sum((q - p) * (q - p), axis=1))

    # define optimization
    def SGD_Momentum(params, lr):
      for i in range(len(self.params)):
        self.moments[i][:] = self.momentum * self.moments[i] + \
            lr * self.params[i].grad
        self.params[i][:] = self.params[i] - self.moments[i]

    self.nn = fc_architecture
    self.mean_loss = sqerr
    self.optimizer = SGD_Momentum

  def preprocessBatching(self, x, pad=True):
    if len(x.shape) == 1:
      x = x.reshape((1, -1))
    if pad:
      datalen = ((x.shape[0] - 1) // self.batch_size + 1) * self.batch_size
      x = np.concatenate([x, np.zeros([datalen - x.shape[0]] +
        list(x.shape[1:]))], axis=0)
    return x

  def fit(self, dataset, num_epochs=1):
    qstates = self.preprocessBatching(dataset["qstates"])
    qvalues = self.preprocessBatching(dataset["qvalues"])
    idx = list(range(qstates.shape[0]))
    random.shuffle(idx)
    num_batches = len(idx) // self.batch_size

    X = nd.array(qstates[idx], ctx=self.ctx)\
        .reshape([num_batches, self.batch_size] + list(qstates.shape[1:]))
    Y = nd.array(qvalues[idx], ctx=self.ctx)\
        .reshape([num_batches, self.batch_size] + list(qvalues.shape[1:]))

    alpha = self.alpha
    progressBar = ProgressBar(maxval=num_epochs)
    for e in range(num_epochs):
      if self.progbar:
        progressBar.printProgress(e, prefix="Training Q(s,a)",
            suffix="%s / %s" % (e + 1, num_epochs))
      if ((e + 1) % 100) == 0: # 100 epoch alpha decay
        alpha = alpha / 2.0

      for i in range(num_batches):
        with autograd.record():
          outputs, hidden_states = self.nn(X[i])
          loss = self.mean_loss(outputs, Y[i])
          loss.backward()
        self.optimizer(self.params, alpha)

        # Keep a moving average of the losses
        if (i == 0) and (e == 0):
          self.moving_loss = np.mean(loss.asnumpy()[0])
        else:
          self.moving_loss = 0.99 * self.moving_loss + 0.01 * \
              np.mean(loss.asnumpy()[0])

  def predict(self, qstates):
    lastqlen = qstates.shape[0] % self.batch_size
    qstates = self.preprocessBatching(qstates)
    num_batches = qstates.shape[0] // self.batch_size
    X = nd.array(qstates, ctx=self.ctx)\
        .reshape([num_batches, self.batch_size] + list(qstates.shape[1:]))

    qvalues = []
    for i in range(num_batches):
      outputs, hidden_states = self.nn(X[i])
      Y = outputs.asnumpy()
      if i == num_batches - 1 and lastqlen != 0:
        Y = Y[:lastqlen]
      qvalues.append(Y)
    qvalues = np.concatenate(qvalues, axis=0)
    return qvalues

  def __call__(self, data):
    return self.predict(data)

  def score(self):
    return self.moving_loss

  def load_params(self, params_filename):
    if os.path.isfile(params_filename):
      self.Wdense1, self.bdense1, \
      self.Wdense2, self.bdense2, \
      self.Wdense3, self.bdense3, \
      self.Wdense4, self.bdense4, \
      self.Wdense5, self.bdense5 \
        = nd.load(params_filename)
      self.params = [
          self.Wdense1, self.bdense1,
          self.Wdense2, self.bdense2,
          self.Wdense3, self.bdense3,
          self.Wdense4, self.bdense4,
          self.Wdense5, self.bdense5
          ]
      for param in self.params:
        param.attach_grad()
      self.moments = [nd.zeros(param.shape, ctx=self.ctx)
          for param in self.params]

  def save_params(self, params_filename):
    nd.save(params_filename, [
      self.Wdense1, self.bdense1,
      self.Wdense2, self.bdense2,
      self.Wdense3, self.bdense3,
      self.Wdense4, self.bdense4,
      self.Wdense5, self.bdense5
      ])

class PoWERDistribution:
  def __init__(self, n_states, n_actions, sigma=1.0):
    self.theta = np.random.random([n_states, n_actions])
    self.sigma = np.ones([n_states, n_actions], dtype=np.float64) * sigma
    self.eps = None
    self.error = 0

  def predict(self, currentState, dataset): # sample
    vectored = False
    if len(currentState.shape) == 1:
      currentState = np.array([currentState])
      vectored = True
    self.eps = np.reshape(np.random.normal(scale=self.sigma.flatten()),
        self.theta.shape)
    Theta = self.theta + self.eps
    s = dataset["states"]
    s_t = repmat(currentState, max(s.shape[0], 1), 1)

    def RBF(dx):
      alpha = 1.0
      return np.sum(np.exp(-alpha * np.multiply(dx, dx)), axis=0) / \
          float(dx.shape[0])

    phi = RBF(s_t - s) if s.shape[0] > 0 else np.zeros(s_t.shape)
    a = np.dot(Theta.T, phi.T)
    if vectored:
      a = a.flatten()
    return a.copy(), self.eps.copy()

  def fit(self, dataset):
    N = len(dataset["values"])
    weightedq = np.sum([dataset["values"][i] * dataset["info"][i]["eps"]
      for i in range(N)], axis=0)
    totalq = sum([dataset["values"][i] for i in range(N)]) + 0.00001
    update = weightedq / totalq
    self.error = np.sum(np.square(update))
    self.theta += update

  def score(self):
    return self.error

  def clear(self):
    self.dataset = []

  def load_params(self, params_filename):
    self.theta = np.load(params_filename)

  def save_params(self, params_filename):
    np.save(params_filename, self.theta)
