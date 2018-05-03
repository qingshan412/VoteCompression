"""Simple classifiers + voter -> new classes
"""
import numpy as np
import tensorflow as tf

# def levels_init(meta_number=10, level_number = 2):
#   with np.load(os.path.join(FLAGS.data_dir,FLAGS.dataset,
#         "ground_train.npz")) as data:
#     features = data["features"]

#   levels = np.random.randint(0, level_number, 
#         size=[features.shape[0],1])#.astype('int64')  
#   np.savez("levels.npz", levels = levels)

class voter(object):
  def __init__(self, data_size, feature_size, level_number=2):
    self.levels = np.random.randint(0, level_number, size=[data_size,1])
    # .astype('int64')  
    # np.savez("levels.npz", levels = levels)
    self.estimator_list=[]

    # vote0
    self.estimator_list.append(tf.contrib.kernel_methods.KernelLinearClassifier(
          feature_columns=[, ],
          model_dir="./voters/0",
          n_classes=level_number))
    # vote1
    kernel_mappers1 = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
          input_dim=feature_size,
          output_dim=feature_size*2)
    self.estimator_list.append(tf.contrib.kernel_methods.KernelLinearClassifier(
          feature_columns=[, ],
          model_dir="./voters/1",
          n_classes=level_number,
          kernel_mappers=kernel_mappers1))
    # vote2
    kernel_mappers2 = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
          input_dim=feature_size,
          output_dim=int(feature_size/2),
          stddev=)
    self.estimator_list.append(tf.contrib.kernel_methods.KernelLinearClassifier(
          feature_columns=[, ],
          model_dir="./voters/2",
          n_classes=level_number,
          kernel_mappers=kernel_mappers2))
    #vote3
    self.estimator_list.append(tf.estimator.LinearClassifier(
          feature_columns=[,],
          model_dir="./voters/lc",
          n_classes=level_number))
    #vote4
    self.estimator_list.append(tf.estimator.BaselineClassifier(
          model_dir="./voters/bc",
          n_classes=level_number))

  def vote2level(self):
    ''' self.votes: [data_size, vote_size]
    update self.levels: [data_size, 1]'''
    for i in range(self.votes.shape[0]):
      self.levels[i, 0] = np.argmax(np.bincount(self.votes[i,:]))

  def upate_vote(self, images):
    '''update self.votes'''
    column[0] = self.estimator_list[0].fit(image)
    for i in range(len(self.estimator_list)-1):
        column.append(self.estimator_list[i+1].fit(image))
    votes = tf.stack(column)
    
class bp_voter(object):
  def __init__(self, data_size, feature_size, level_number=2):
    self.levels = np.random.randint(0, level_number, size=[data_size,1])
    # .astype('int64')  
    # np.savez("levels.npz", levels = levels)
    self.estimator_list=[]

    # vote0
    self.estimator_list.append(tf.estimator.LinearClassifier(
          feature_columns=[,],
          model_dir="./voters/lc0",
          n_classes=level_number))
    # vote1
    self.estimator_list.append(tf.estimator.LinearClassifier(
          feature_columns=[,],
          model_dir="./voters/lc1",
          n_classes=level_number))
    # vote2
    self.estimator_list.append(tf.estimator.LinearClassifier(
          feature_columns=[,],
          model_dir="./voters/lc2",
          n_classes=level_number))
    
  def vote2level(self):
    ''' self.votes: [data_size, vote_size]
    update self.levels: [data_size, 1]'''
    for i in range(self.votes.shape[0]):
      self.levels[i, 0] = np.argmax(np.bincount(self.votes[i,:]))

  def upate_vote(self, images):
    '''update self.votes'''
    


class train_input(object):
  def __init__(self, sess, mode, batch_size):
    self.sess = sess

    if mode == "train":
      with np.load(os.path.join(FLAGS.data_dir,FLAGS.dataset,
            "ground_train.npz")) as data:
        features = data["features"]
        labels = data["labels"]
      levels = np.random.randint(0,2,size=[features.shape[0],1])
      # np.savez("levels.npz", levels = levels)
      levels_placeholder = tf.placeholder(levels.dtype, levels.shape)
    else:
      with np.load(os.path.join(FLAGS.data_dir,FLAGS.dataset,
            "ground_test.npz")) as data:
        features = data["features"]
        labels = data["labels"]

    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    if mode == "train":
      dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, 
            levels_placeholder, labels_placeholder))
    else:
      dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, 
            labels_placeholder))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(tf.constant(features.shape[0], dtype=tf.int64)
            ).repeat().batch(batch_size)
    # iterator
    self.iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                            dataset.output_shapes)
    # Return the read end of the pipeline.
    # run_config = tf.ConfigProto()
    # run_config.gpu_options.allow_growth=True
    # with tf.Session(config=run_config) as sess:
    self.sess.run(self.iterator.make_initializer(dataset), 
            feed_dict={features_placeholder: features, levels_placeholder: levels})
  
  # @property
  def next_value(self):
    return self.sess.run(self.iterator.get_next())
  
  def test_input_fn(self):
      return tf.estimator.inputs.numpy_input_fn(
            x=self.features,#
            y=self.labels,#
            batch_size=128,
            num_epochs=1,
            shuffle=None,
            queue_capacity=1000,
            num_threads=1)

def my_model_fn_level_1(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode,     # An instance of tf.estimator.ModeKeys
    params):  # Additional configuration
  
  
  # Input Layer
  # Reshape images to 4-D tensor: [batch_size, width, height, channels]
  image_size = params['image_size']
  batch_size = params['batch_size']
  num_class = params['num_class']
  lr = params['lr']
  input_layer = tf.reshape(features["images"], [-1, image_size, image_size, 3])

  conv1 = tf.layers.conv2d(inputs=input_layer, filters=64,
        kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(conv1, pool_size=[3,3], strides=[2,2], padding='same')
  norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    
  conv2 = tf.layers.conv2d(inputs=norm1, filters=64, kernel_size=[5, 5],
        padding="same", activation=tf.nn.relu)
  norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)
  pool2 = tf.layers.max_pooling2d(norm2, pool_size=[3,3], strides=[2,2], padding='same')
  
  local3 = tf.layers.dense(tf.reshape(pool2, [batch_size,-1]), units=384, activation=tf.nn.relu)
  logits = tf.layers.dense(local3, units=num_class, activation=tf.nn.relu)

  # Compute predictions.
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          'class_ids': predicted_classes[:, tf.newaxis],
          'probabilities': tf.nn.softmax(logits),
          'logits': logits,
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)


  # Compute loss.
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Compute evaluation metrics.
  accuracy = tf.metrics.accuracy(labels=labels,
                                 predictions=predicted_classes,
                                 name='acc_op')
  metrics = {'accuracy': accuracy}
  tf.summary.scalar('accuracy', accuracy[1])

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

  # Create training op.
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

