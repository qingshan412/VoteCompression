"""Simple classifiers + voter -> new classes
"""
import numpy as np
import tensorflow as tf

def levels_init(meta_number=10, level_number = 2):
  with np.load(os.path.join(FLAGS.data_dir,FLAGS.dataset,
        "ground_train.npz")) as data:
    features = data["features"]

  levels = np.random.randint(0,2,size=[features.shape[0],1])#.astype('int64')  
  np.savez("levels.npz", levels = levels)

class voter(object):
  def __init__(self):
    x

class train_input(object):
  def __init__(self, sess):
    self.sess = sess

    with np.load(os.path.join(FLAGS.data_dir,FLAGS.dataset,
        "ground_train.npz")) as data:
    features = data["features"]
    levels = np.random.randint(0,2,size=[features.shape[0],1])#.astype('int64')  
    # np.savez("levels.npz", levels = levels)

    features_placeholder = tf.placeholder(features.dtype, features.shape)
    levels_placeholder = tf.placeholder(levels.dtype, levels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, levels_placeholder))
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

def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration
  
  net = tf.feature_column.input_layer(features)
  # Compute loss. 
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
  if mode == tf.estimator.ModeKeys.TRAIN:
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
  # Compute evaluation metrics.
   accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')
  metrics = {'accuracy': accuracy}
  tf.summary.scalar('accuracy', accuracy[1])

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=metrics)
#   run_config = tf.ConfigProto()
#   run_config.gpu_options.allow_growth=True
#   with tf.Session(config=run_config) as sess: