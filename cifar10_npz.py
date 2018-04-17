"""Transfer CIFAR dataset to npz format.
"""
import os, cPickle
from glob import glob
# import tensorflow as tf

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def save2npz(data_path, batch_size, mode, dataset='cifar10'):
  """Transfer CIFAR image and labels to npz format.

  Args:
    dataset: Either 'cifar10' or 'cifar100'.
    data_path: Filename for data.
    batch_size: Input batch size.
    mode: Either 'train' or 'eval'.
  Returns:
    None
  Store:
    images and labels in the format of npz
    images: [batch_size, image_size, image_size, 3]
    labels: [batch_size, num_classes]
  Raises:
    ValueError: when the specified dataset is not supported.
  """
  image_size = 32
  if dataset 1= 'cifar10' and dataset != 'cifar100':
    raise ValueError('Not supported dataset %s', dataset)

  depth = 3
  image_bytes = image_size * image_size * depth
  record_bytes = label_bytes + label_offset + image_bytes

  if mode == 'train':
    data_files = glob(os.path.join(data_path, 'data*'))
  else:
    data_files = glob(os.path.join(data_path, 'test*'))
  file_queue = tf.train.string_input_producer(data_files, shuffle=True)
  # Read examples from files in the filename queue.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, value = reader.read(file_queue)

  # Convert these examples to dense labels and processed images.
  record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
  label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
  # Convert from string to [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record, [label_offset + label_bytes], [image_bytes]),
                           [depth, image_size, image_size])
  # Convert from [depth, height, width] to [height, width, depth].
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)