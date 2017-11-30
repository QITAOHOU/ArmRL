import numpy as np
import mxnet as mx

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
      "learning_rate": self.alpha
      })

  def preprocessBatching(self, x):
    if len(x.shape) == 1:
      x = np.array([x])
    buflen = self.batch_size - x.shape[0] % self.batch_size
    if buflen < self.batch_size:
      x = np.concatenate([x, np.zeros([buflen, x.shape[1]], dtype=np.float32)])
    return x

  def fit(self, dataset, num_epochs=1):
    if dataset["qvalues"].size == 0:
      return
    train_iter = mx.io.NDArrayIter(
        data=self.preprocessBatching(dataset["qstates"]),
        label=self.preprocessBatching(dataset["qvalues"]),
        batch_size=self.batch_size, shuffle=True,
        data_name="qstate", label_name="qvalue")
    for epoch in range(num_epochs):
      train_iter.reset()
      for batch in train_iter:
        self.model.forward(batch, is_train=True)
        self.model.backward()
        self.model.update()

  def predict(self, qstate):
    qstates = mx.io.NDArrayIter(
        data=self.preprocessBatching(qstate), batch_size=self.batch_size,
        data_name="qstate", label_name="qvalue")
    return [QV.asnumpy() for QV in self.model.predict(qstates)]

  def __call__(self, qstate):
    return self.predict(qstate)
