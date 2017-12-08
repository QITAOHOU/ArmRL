import numpy as np
from numpy.matlib import repmat
import mxnet as mx
import os
import memory

class MxFullyConnected:
  def __init__(self, sizes=[1, 1], batch_size=32, alpha=0.01, use_gpu=False):
    assert(len(sizes) >= 2)
    self.input_size = sizes[0]
    self.output_size = sizes[-1]
    self.batch_size = batch_size
    self.layer_sizes = []
    if len(sizes) > 2:
      self.layer_sizes = sizes[1:-1]
    self.alpha = alpha
    self.avg_error = 0.0

    # define feeds
    self.x = mx.sym.Variable("data")
    self.y = mx.sym.Variable("label")

    # define memory
    self.w = []
    self.b = []
    for i in range(len(self.layer_sizes)):
      self.w.append(mx.sym.Variable("l" + str(i) + "_w",
        init=mx.init.Normal(0.1)))
      self.b.append(mx.sym.Variable("l" + str(i) + "_b",
        init=mx.init.Constant(0.1)))
    self.w.append(mx.sym.Variable("out_w", init=mx.init.Normal(0.1)))
    self.b.append(mx.sym.Variable("out_b", init=mx.init.Constant(0.1)))

    # define architecture
    self.fc = []
    self.relu = []
    lastlayer = self.x
    for i in range(len(self.layer_sizes)):
      self.fc.append(mx.sym.FullyConnected(data=lastlayer, weight=self.w[i],
        bias=self.b[i], num_hidden=self.layer_sizes[i], name="fc" + str(i)))
      self.relu.append(mx.sym.Activation(data=self.fc[-1], act_type='relu',
        name="relu" + str(i)))
      lastlayer = self.relu[i]
    self.fc.append(mx.sym.FullyConnected(data=lastlayer, weight=self.w[-1],
      bias=self.b[-1], num_hidden=self.output_size, name="out"))
    self.y_ = mx.sym.LinearRegressionOutput(data=self.fc[-1], label=self.y,
        name="loss")

    # define training
    if use_gpu:
      self.model = mx.mod.Module(self.y_, context=mx.gpu(0),
          data_names=["data"], label_names=["label"])
    else:
      self.model = mx.mod.Module(self.y_, context=mx.cpu(0), # start w/ cpu 4now
          data_names=["data"], label_names=["label"])
    self.model.bind(
        data_shapes=[("data", (self.batch_size, self.input_size))],
        label_shapes=[("label", (self.batch_size, self.output_size))])
    self.model.init_params()
    self.model.init_optimizer(optimizer="adam", optimizer_params={
      "learning_rate": self.alpha,
      "wd": 0.01
      })

  def preprocessBatching(self, x):
    if len(x.shape) == 1:
      x = np.array([x])
    buflen = self.batch_size - x.shape[0] % self.batch_size
    if buflen < self.batch_size:
      x = np.concatenate([x, np.zeros([buflen, x.shape[1]], dtype=np.float32)])
    return x

  def fit(self, dataset, num_epochs=1):
    data = dataset["data"]
    label = dataset["label"]
    if label.size == 0:
      return
    if len(data.shape) == 2 and len(label.shape) == 1:
      label = np.array([label]).T
    train_iter = mx.io.NDArrayIter(
        data=self.preprocessBatching(data),
        label=self.preprocessBatching(label),
        batch_size=self.batch_size, shuffle=True,
        data_name="data", label_name="label")
    error = mx.metric.MSE()
    total_error = 0.0
    for epoch in range(num_epochs):
      train_iter.reset()
      error.reset()
      for batch in train_iter:
        self.model.forward(batch, is_train=True)
        self.model.update_metric(error, batch.label)
        self.model.backward()
        self.model.update()
      total_error += error.get()[1]
    self.avg_error = total_error / num_epochs

  def predict(self, data):
    data_iter = mx.io.NDArrayIter(
        data=self.preprocessBatching(data), batch_size=self.batch_size,
        data_name="data", label_name="label")
    return np.array([QV.asnumpy()
      for QV in self.model.predict(data_iter)])[:data.shape[0], :]

  def __call__(self, data):
    return self.predict(data)

  def score(self):
    return self.avg_error

  def load_params(self, params_filename):
    if os.path.isfile(params_filename):
      self.model.load_params(params_filename)

  def save_params(self, params_filename):
    self.model.save_params(params_filename)

def RBF(s_t, s):
  alpha = 1.0
  return np.sum(np.exp(-alpha * np.multiply(s_t - s, s_t - s)), axis=0) / \
      float(s.shape[0])

class PoWERDistribution:
  def __init__(self, n_states, n_actions, sigma=1.0):
    self.theta = np.random.random([n_states, n_actions])
    #self.sigma = np.random.random([n_states, n_actions])
    self.sigma = np.ones([n_states, n_actions], dtype=np.float32) * sigma
    self.dataset = []
    self.eps = None
    self.error = 0

  def predict(self, currentState): # sample
    vectored = False
    if len(currentState.shape) == 1:
      currentState = np.array([currentState])
      vectored = True
    self.eps = np.random.normal(scale=self.sigma.flatten())
    W = self.theta + np.reshape(self.eps, self.theta.shape)
    phi = np.array([
      RBF(currentState, np.array([x["state"] for x in self.dataset]))]) \
        if len(self.dataset) > 0 else np.zeros(currentState.shape)
    a = np.dot(W.T, phi.T)
    if vectored:
      a = a.flatten()
    return a

  def append(self, state, action, nextState, reward):
    self.dataset.append({
      "state": state,
      "action": action,
      "nextState": nextState,
      "reward": reward,
      "eps": self.eps
      })

  def fit(self):
    dataset = memory.Bellman(self.dataset, 1.0)
    weightedq = np.sum([x["value"] * x["eps"] for x in dataset], axis=0)
    totalq = sum([x["value"] for x in dataset])
    if totalq == 0.0:
      self.error = 0
      return
    update = np.reshape(weightedq / totalq, self.theta.shape)
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

class ActorCritic:
  def __init__(self):
    pass
