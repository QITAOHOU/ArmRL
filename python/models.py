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
    self.x = mx.sym.Variable("qstate")
    self.y = mx.sym.Variable("qvalue")

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
          data_names=["qstate"], label_names=["qvalue"])
    else:
      self.model = mx.mod.Module(self.y_, context=mx.cpu(0), # start w/ cpu 4now
          data_names=["qstate"], label_names=["qvalue"])
    self.model.bind(
        data_shapes=[("qstate", (self.batch_size, self.input_size))],
        label_shapes=[("qvalue", (self.batch_size, self.output_size))])
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
    qstates = dataset["qstates"]
    qvalues = dataset["qvalues"]
    if qvalues.size == 0:
      return
    if len(qstates.shape) == 2 and len(qvalues.shape) == 1:
      qvalues = np.array([qvalues]).T
    train_iter = mx.io.NDArrayIter(
        data=self.preprocessBatching(qstates),
        label=self.preprocessBatching(qvalues),
        batch_size=self.batch_size, shuffle=True,
        data_name="qstate", label_name="qvalue")
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

  def predict(self, qstate):
    qstates = mx.io.NDArrayIter(
        data=self.preprocessBatching(qstate), batch_size=self.batch_size,
        data_name="qstate", label_name="qvalue")
    return np.array([QV.asnumpy()
      for QV in self.model.predict(qstates)])[:qstate.shape[0], :]

  def __call__(self, qstate):
    return self.predict(qstate)

  def score(self):
    return self.avg_error

  def load_params(self, params_filename):
    if os.path.isfile(params_filename):
      self.model.load_params(params_filename)

  def save_params(self, params_filename):
    self.model.save_params(params_filename)

class PoWERDistribution:
  def __init__(self, n_states, n_actions, sigma=1.0):
    self.theta = np.random.random([n_states, n_actions])
    #self.sigma = np.random.random([n_states, n_actions])
    self.sigma = np.ones([n_states, n_actions], dtype=np.float32) * sigma
    self.dataset = []
    self.eps = None

  def predict(self, currentState):
    vectored = False
    if len(currentState.shape) == 1:
      currentState = np.array([currentState])
      vectored = True
    num_items = 1
    alpha = 1.0
    s = np.array([x["state"] for x in self.dataset])
    print(currentState.shape, s.shape)
    RBF = lambda s_t: np.exp(-alpha * np.dot((s_t - s).T, s_t - s)) / s.shape[0]
    self.eps = np.random.normal(
        scale=repmat(self.sigma.flatten(), num_items, 1))
    W = self.theta + np.reshape(self.eps[0, :], self.theta.shape)
    if len(self.dataset) == 0:
      print("zeros")
      a = np.dot(W.T, np.zeros(currentState.shape).T)
    else:
      print("power")
      print("Wshape", W.T.shape)
      a = np.dot(W.T, RBF(currentState))
      print("OLDASHAPE", a.shape)
    if vectored:
      a = a.flatten()
    print("ASHAPE", a.shape)
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
    qeps = [np.multiply(x["value"], x["eps"]) for x in dataset]
    qvalues = [x["value"] for x in dataset]
    self.theta = self.theta + np.divide(qeps, qvalues)

  def clear(self):
    self.dataset = []

  def load_params(self, params_filename):
    self.theta = np.load(params_filename)

  def save_params(self, params_filename):
    np.save(params_filename, self.theta)

#class Actor:
#  def __init__(self):
#    # TODO: add neural net to estimate p(a|V,s)

#class Critic:
#  def __init__(self):
#    # TODO: add neural net to estimate p(V|s,a)
