import numpy as np
from numpy.matlib import repmat
import mxnet as mx
from mxnet import nd, autograd
import os
import random
from console_widgets import Progbar

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

    # temp not used:
    self.Wrnn_h1 = nd.random_normal(shape=(s[2], s[3]), ctx=self.ctx) * 0.01
    #self.W_h3h3  = nd.random_normal(shape=(s[3], s[3]), ctx=self.ctx) * 0.01
    self.brnn_1  = nd.random_normal(shape=(s[3]), ctx=self.ctx) * 0.01

    conv1_shape  = (s[3], 1, 3) # out, in, ksize
    self.Wconv1  = nd.random_normal(shape=conv1_shape, ctx=self.ctx) * 0.01
    self.bconv1  = nd.random_normal(shape=(s[3]), ctx=self.ctx) * 0.01
    conv1out     = s[3] * (s[2] - 2)

    self.Wdense3 = nd.random_normal(shape=(s[2], s[4]), ctx=self.ctx) * 0.01
    self.bdense3 = nd.random_normal(shape=(s[4]), ctx=self.ctx) * 0.01

    self.Wdense4 = nd.random_normal(shape=(s[4], s[5]), ctx=self.ctx) * 0.01
    self.bdense4 = nd.random_normal(shape=(s[5]), ctx=self.ctx) * 0.01

    self.Wdense5 = nd.random_normal(shape=(s[5], s[6]), ctx=self.ctx) * 0.01
    self.bdense5 = nd.random_normal(shape=(s[6]), ctx=self.ctx) * 0.01

    self.params = [
        self.Wdense1, self.bdense1,
        self.Wdense2, self.bdense2,
        #self.Wconv1,  self.bconv1,
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

    def conv_architecture(inputs, state={}):
      dense1 = nd.relu(nd.dot(inputs, self.Wdense1) + self.bdense1)
      dense2 = nd.relu(nd.dot(dense1, self.Wdense2) + self.bdense2)
      frame1 = nd.reshape(dense2, [dense2.shape[0], 1] + list(dense2.shape[1:]))
      conv1  = nd.relu(nd.Convolution(frame1, self.Wconv1, self.bconv1,
        kernel=(3,), num_filter=s[3]))
      flat1  = nd.reshape(conv1, (conv1.shape[0], -1))
      dense3 = nd.relu(nd.dot(flat1, self.Wdense3) + self.bdense3)
      dense4 = nd.relu(nd.dot(dense3, self.Wdense4) + self.bdense4)
      qvalues = nd.dot(dense4, self.Wdense5) + self.bdense5
      return (qvalues, {})

    def rnn_architecture(inputs, state={}):
      #outputs = []
      rnn1_h = state["rnn1_h"] if "rnn1_h" in state else None
      if type(rnn1_h) == type(None):
        input_len = inputs.shape[1]
        rnn1_h = nd.zeros(shape=(input_len, self.sizes[3]), ctx=self.ctx)
      for x in inputs:
        dense1 = nd.relu(nd.dot(x, self.W_xh1) + self.b_h1)
        dense2 = nd.relu(nd.dot(dense1, self.W_h1h2) + self.b_h2)
        rnn1_h = nd.tanh(nd.dot(dense2, self.W_h2h3) + \
            nd.dot(rnn1_h, self.W_h3h3) + self.b_h3)
        #outputs.append(rnn1_h)
      dense3 = nd.relu(nd.dot(rnn1_h, self.W_h3h4) + self.b_h4)
      dense4 = nd.relu(nd.dot(dense3, self.W_h4h5) + self.b_h5)
      qvalues = nd.dot(dense4, self.W_h5y_) + self.b_y_
      return (qvalues, {"rnn1_h": rnn1_h})

    # define loss
    def sqerr(p, q): # p is the real distribution, q is the predicted
      return nd.mean(nd.sum((q - p) * (q - p), axis=1)) #, exclude=True))

    def avg_loss(outputs, labels):
      assert(len(outputs) == len(labels))
      total_loss = 0.0
      for (output, label) in zip(outputs, labels):
        total_loss = total_loss + sqerr(output, label)
      return total_loss / len(outputs)

    def SGD_Momentum(params, lr):
      for i in range(len(self.params)):
        self.moments[i][:] = self.momentum * self.moments[i] + \
            lr * self.params[i].grad
        self.params[i][:] = self.params[i] - self.moments[i]

    self.nn = fc_architecture
    #self.loss = sqerr
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
        .reshape([num_batches, self.batch_size] + list(qstates.shape[1:]))\
        #.swapaxes(1, 2)
    X = X.reshape(list(X.shape)[:-1])
    Y = nd.array(qvalues[idx], ctx=self.ctx)\
        .reshape([num_batches, self.batch_size] + list(qvalues.shape[1:]))

    alpha = self.alpha
    progressBar = Progbar(maxval=num_epochs, prefix="Training Q(s,a)")
    for e in range(num_epochs):
      if self.progbar:
        progressBar.printProgress(e)
      if ((e + 1) % 100) == 0:
        alpha = alpha / 2.0

      #hidden_states = {"rnn1_h": 
      #    nd.zeros(shape=(self.batch_size, self.sizes[4]), ctx=self.ctx)}
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
        #.swapaxes(1, 2)
    X = X.reshape(list(X.shape)[:-1])

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

def RBF(dx):
  alpha = 1.0
  return np.sum(np.exp(-alpha * np.multiply(dx, dx)), axis=0) / \
      float(dx.shape[0])

class PoWERDistribution:
  def __init__(self, n_states, n_actions, sigma=1.0):
    self.theta = np.random.random([n_states, n_actions])
    #self.sigma = np.random.random([n_states, n_actions])
    self.sigma = np.ones([n_states, n_actions], dtype=np.float64) * sigma
    self.eps = None
    self.error = 0

  def predict(self, currentState, dataset): # sample
    vectored = False
    if len(currentState.shape) == 1:
      currentState = np.array([currentState])
      vectored = True
    self.eps = np.random.normal(scale=self.sigma.flatten())
    Theta = self.theta + np.reshape(self.eps, self.theta.shape)
    s = dataset["states"]
    s_t = repmat(currentState, max(s.shape[0], 1), 1)
    phi = RBF(s_t - s) if s.shape[0] > 0 else np.zeros(s_t.shape)
    a = np.dot(Theta.T, phi.T)
    if vectored:
      a = a.flatten()
    return a.copy(), self.eps.copy()

  def fit(self, dataset):
    weightedq = np.sum([dataset["values"][i] * dataset["info"][i]["eps"]
      for i in range(len(dataset))], axis=0)
    totalq = sum([dataset["values"][i] for i in range(len(dataset))]) + 0.00001
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
      x = np.concatenate([x, np.zeros([buflen, x.shape[1]], dtype=np.float64)])
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
