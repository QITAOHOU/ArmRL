import tensorflow as tf

class FullyConnected:
  def __init__(self, sizes=[1, 1], alpha=0.1):
    assert(len(sizes) >= 2)
    self.input_size = sizes[0]
    self.output_size = sizes[-1]
    self.batch_size = 32
    self.layer_sizes = []
    if len(sizes) > 2:
      self.layer_sizes = sizes[1:]
    self.alpha = alpha

    # placeholders
    self.x = tf.placeholder(tf.float32, [None, self.input_size],
        name="stateobs")
    self.y = tf.placeholder(tf.float32, [None, self.output_size],
        name="qvalue")

    # variables
    self.w = []
    self.b = []
    interim_size = self.input_size
    for i in range(len(self.layer_sizes)):
      self.w.append(tf.Variable(tf.truncated_normal(
        [interim_size, self.layer_sizes[i]], stddev=0.1),
        name="l" + str(i) + "w"))
      self.b.append(tf.Variable(tf.constant(0.1, shape=[self.layer_sizes[i]]),
        name="l" + str(i) + "b"))
      interim_size = self.layer_sizes[i]

    # layers
    self.fc = []
    lastlayer = self.x
    for i in range(len(self.layer_sizes)):
      self.fc.append(tf.nn.relu(tf.matmul(lastlayer, self.w[i]) + self.b[i]),
        name="fc" + str(i))
      lastlayer = self.fc[i]

    # training params
    self.loss = tf.reduce_mean(
      tf.reduce_sum(tf.square(self.y - self.fc[-1]), reduction_indices=[1]))
    self.train_step = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)

  def train(self, dataset, num_epochs=1):
    assert(dataset.shape[0] % self.batch_size == 0)
    idx = list(range(0, dataset.shape[0]))
    random.shuffle(idx)
    num_batches = dataset.shape[0] / self.batch_size
    batch_id = 0
    for i in range(num_batches):
      new_batch_id = batch_id + self.batch_size
      qstate = dataset["qstates"][idx[batch_id:new_batch_id], :]
      qvalue = dataset["qvalues"][idx[batch_id:new_batch_id], :]
      tf.get_default_session().run(self.train_step, feed_dict={
        self.x: qstate,
        self.y: qvalue
      })

  def __call__(self, qstate):
    return tf.get_default_session().run(self.fc[-1], feed_dict={
      self.x: qstate
    })
