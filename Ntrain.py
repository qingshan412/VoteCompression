"""Non-adversarial Main Process"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

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

T = 10000 #initial temperature
A = 10 #coefficient for stay prob when correct (e^(-A*T))
B = 100 #coefficient for stay prob when when wrong (e^(-B*T))

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
        level_labels.append(labels[pos,:])
    return level_images, level_labels

# def model_assign():
#     level_infers_op = []
#     level_losses_op = []
#     for i in range(FLAGS.level_number):
#         level_infers_op.append(Dmodels.infer(i))
#         level_losses_op.append(Dmodels.loss(i))
#     return level_infers_op, level_losses_op

def main(_):
    np.random.seed(22)

    levels = level_init()
    with np.load(os.path.join(FLAGS.data_dir,FLAGS.dataset,
            "ground_train.npz")) as data:
        images = data["features"]
        labels =  data["labels"]

    level_images, level_labels = level_assign(images, labels, levels)
    
    level_opt_op_0 = tf.train.AdamOptimizer(learning_rate=0.1)
    level_opt_op_1 = tf.train.AdamOptimizer(learning_rate=0.01)
    level_opt_op_2 = tf.train.AdamOptimizer(learning_rate=0.01)
    level_opt_ops = [level_opt_op_0, level_opt_op_1, level_opt_op_2]

    level_logits = []
    #level_logit_values = []
    level_losses = []
    level_loss_values = []
    level_opts = []
    for i in range(FLAGS.level_number):
        level_logits.append(0)
        #level_logit_values.append(0)
        level_losses.append(0)
        level_loss_values.append(0)
        level_opts.append(0)
    

    for i in range(FLAGS.level_number):
        # train for every model
        if level_images[i]:
            # train model[i] by level_images[i]
            # output = inference[i]
            if i == 0:
                level_logits[i] = Dmodels.level_infers_op_0(level_images[i])
                level_losses[i] = Dmodels.level_losses_op_0(level_labels[i], level_logits[i])
            elif i == 1:
                level_logits[i] = Dmodels.level_infers_op_1(level_images[i])
                level_losses[i] = Dmodels.level_losses_op_1(level_labels[i], level_logits[i])
            elif i == 2:
                level_logits[i] = Dmodels.level_infers_op_2(level_images[i])
                level_losses[i] = Dmodels.level_losses_op_2(level_labels[i], level_logits[i])
            else:
                print("unsupported level "+ str(i)+"!")

            level_opts[i] = level_opt_ops[i].minimize(level_losses[i])
                
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    sess.run(init)


    for it in range(FLAGS.max_iter):
        # iteration
        for i in range(FLAGS.level_number):
            _, loss_t, logit_t = ess.run([level_opts[i], level_losses[i], level_logits[i]])
            level_loss_values[i] = loss_t
            # level_logit_values[i] = logit_t
            # test accuracy
            # xxx
            # print

            # move samples
            output = np.argmax(logit_t, 1)
            for j in range(output.shape[0]):
                if output[j] == level_labels[i][j]:
                    if np.random.random_sample() > np.exp(-A*T):
                        if i<1:
                            level_image[i+1].append(level_image[i].pop(j))
                            level_labels[i+1].append(level_labels[i].pop(j))
                        elif i>(FLAGS.level_number-2):
                            level_image[i-1].append(level_image[i].pop(j))
                            level_labels[i-1].append(level_labels[i].pop(j))
                        else:
                            if np.random.random_sample() < 0.5:
                                level_image[i+1].append(level_image[i].pop(j))
                                level_labels[i+1].append(level_labels[i].pop(j))
                            else:
                                level_image[i-1].append(level_image[i].pop(j))
                                level_labels[i-1].append(level_labels[i].pop(j))
                else:
                    if np.random.random_sample() > np.exp(-B*T):
                        if i<1:
                            level_image[i+1].append(level_image[i].pop(j))
                            level_labels[i+1].append(level_labels[i].pop(j))
                        elif i>(FLAGS.level_number-2):
                            level_image[i-1].append(level_image[i].pop(j))
                            level_labels[i-1].append(level_labels[i].pop(j))
                        else:
                            if np.random.random_sample() < 0.5:
                                level_image[i+1].append(level_image[i].pop(j))
                                level_labels[i+1].append(level_labels[i].pop(j))
                            else:
                                level_image[i-1].append(level_image[i].pop(j))
                                level_labels[i-1].append(level_labels[i].pop(j))
    
    


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)