"""Non-adversarial Main Process"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import Dmodels

# tf.logging.set_verbosity(tf.logging.INFO)
# import argparse

flags = tf.app.flags
flags.DEFINE_integer("level_number", 3, "Number of different classifiers [3]")
flags.DEFINE_integer("image_size", 32, "The size of images [32]")
flags.DEFINE_integer("max_iter", 10, "Max number of iterations [10]")
flags.DEFINE_string("data_dir", "../../../data", "Directory to datasets [../../../data]")
flags.DEFINE_string("dataset", "cifar10", "The name of dataset [cifar10]")

flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")

FLAGS = flags.FLAGS

def level_init():
    with np.load(os.path.join(FLAGS.data_dir,FLAGS.dataset,
            "ground_train.npz")) as data:
        # images = data["features"]
        labels = data["labels"]
    levels = np.random.randint(0,FLAGS.level_number,size=[labels.shape[0],1])
    return levels

def level_assign(images, labels, levels):
    level_images = []
    level_labels = []
    for i in range(FLAGS.level_number):
        pos = np.argwhere(levels==i)
        level_images.append(images[pos,:,:])
        level_lables.append(labels[pos,:])
    return level_images, level_labels

def model_assign():
    level_infers_op = []
    level_losses_op = []
    for i in range(FLAGS.level_number):
        level_infers_op.append(Dmodels.infer(i))
        level_losses_op.append(Dmodels.loss(i))
    return level_infers_op, level_losses_op

def main(_):
    levels = level_init()
    with np.load(os.path.join(FLAGS.data_dir,FLAGS.dataset,
            "ground_train.npz")) as data:
        images = data["features"]
        lablels =  data["labels"]

    level_images, level_labels = level_assign(images, labels, levels)

    level_infers_op, level_losses_op = model_assign()

    level_logits = []
    level_losses = []

    for it in range(FLAGS.max_iter):
        # iteration
        for i in range(FLAGS.level_number):
            # train for every model
            if level_images[i]:
                # train model[i] by level_images[i]
                # output = inference[i]
                level_logits[i] = level_infers_op[i](level_images[i])
                level_losses[i] = level_losses_op[i](level_labels[i], level_logits[i])
                # sess.run()
                
                # test accuracy
                xxx
                print
                # move samples
                for j in range(output.shape[0]):
                    if output[j] == level_labels[i][j]:
                        level_images[i][j]+level_labels[i][j] prob to move/stay
                    else:
                        level_images[i][j]+level_labels[i][j] prob to move/stay
    
    


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)