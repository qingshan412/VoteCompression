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
flags.DEFINE_integer("max_iter", 500, "Max number of iterations [10]")
flags.DEFINE_string("data_dir", "../../../data", "Directory to datasets [../../../data]")
flags.DEFINE_string("dataset", "cifar10", "The name of dataset [cifar10]")

flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")

FLAGS = flags.FLAGS

A = 0.05 #coefficient for stay prob when correct (e^(-A*T))
B = 0.01 #coefficient for stay prob when when wrong (e^(-B*T))

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
        pos = np.argwhere(levels==i)[:,0]
        # print('pos...')
        # print(pos.shape)
        # print('assign...')
        # print(images[pos,:,:].shape)
        level_images.append(images[pos,:,:])
        level_labels.append(labels[pos,:])
        level_labels[i]=np.squeeze(level_labels[i]) 
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

    # print('images size:')
    # print(images.shape)
    # print('labels size:')
    # print(labels.shape)
    # print('levels size:')
    # print(levels.shape)
    # print(np.unique(levels))

    level_images, level_labels = level_assign(images, labels, levels)
    #print(level_labels[0])
    # print('level_images:')
    # print(len(level_images))
    # print(level_images[0].shape)
    # exit(0)

    
    level_opt_op_0 = tf.train.AdamOptimizer(learning_rate=0.1)
    level_opt_op_1 = tf.train.AdamOptimizer(learning_rate=0.01)
    level_opt_op_2 = tf.train.AdamOptimizer(learning_rate=0.01)
    level_opt_ops = [level_opt_op_0, level_opt_op_1, level_opt_op_2]

    level_logits = []
    #level_logit_values = []
    level_losses = []
    level_opts = []
    #Y.D.
    level_images_tensor=[]
    level_labels_tensor=[]
    level_loss_values = [[] for i in range(3)]
    level_num_values = [[] for i in range(3)]
    for i in range(FLAGS.level_number):
        level_logits.append(0)
        #level_logit_values.append(0)
        level_losses.append(0)
        level_opts.append(0)
    
	level_images_tensor.append(tf.convert_to_tensor(level_images[i][0:FLAGS.batch_size], dtype=tf.float32))
	level_labels_tensor.append(tf.convert_to_tensor(level_labels[i][0:FLAGS.batch_size], dtype=tf.float32))

    for i in range(FLAGS.level_number):
        # train for every model
        if len(level_images[i])>0:
            # train model[i] by level_images[i]
            # output = inference[i]
            print('level '+str(i)+':')
            print(level_images[i].shape)
            if i == 0:
                level_logits[i] = Dmodels.level_infers_op_0(level_images_tensor[i], FLAGS)
                level_losses[i] = Dmodels.level_losses_op_0(level_labels_tensor[i], level_logits[i])
            elif i == 1:
                level_logits[i] = Dmodels.level_infers_op_1(level_images_tensor[i], FLAGS)
                level_losses[i] = Dmodels.level_losses_op_1(level_labels_tensor[i], level_logits[i])
            elif i == 2:
                level_logits[i] = Dmodels.level_infers_op_2(level_images_tensor[i], FLAGS)
                level_losses[i] = Dmodels.level_losses_op_2(level_labels_tensor[i], level_logits[i])
            else:
                print("unsupported level "+ str(i)+"!")

            level_opts[i] = level_opt_ops[i].minimize(level_losses[i])
                
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    sess = tf.Session()
    sess.run(init)

    steps_per_iter = 500
    pred_result = [[0 for i in range(50000)] for j in range(3)]

    T = 0
    for it in range(FLAGS.max_iter):
	T = T + 1
	print('iter'+str(it))
        for i in range(FLAGS.level_number):
	    print('level'+str(i))
	    print('number of sample:'+str(len(level_images[i])))
	    permu = np.random.permutation(level_images[i].shape[0])
	    level_images[i] = level_images[i][permu]
	    level_labels[i] = level_labels[i][permu]
	    #for level_
	    loss_t = []
	    start_i=0
	    end_i=FLAGS.batch_size
	    for batch_i in range(steps_per_iter):
		    if start_i >= len(level_images[i]):
	    		batch_images = np.array(level_images[i][0:FLAGS.batch_size])
	    		batch_labels = np.array(level_labels[i][0:FLAGS.batch_size])
			start_i = 0
			end_i = FLAGS.batch_size
		    else:
			if end_i > len(level_images[i]):
	    		    batch_images = np.array(level_images[i][start_i:len(level_images[i])])
	    		    batch_labels = np.array(level_labels[i][start_i:len(level_labels[i])])
			    batch_images = np.append(batch_images,level_images[i][0:(end_i-len(level_images[i]))],axis=0)
			    batch_labels = np.append(batch_labels,level_labels[i][0:(end_i-len(level_images[i]))],axis=0)
			    start_i = end_i-len(level_images[i])
			    end_i = start_i + FLAGS.batch_size
			else:
	    		    batch_images = np.array(level_images[i][start_i:end_i])
	    		    batch_labels = np.array(level_labels[i][start_i:end_i])
			    start_i = start_i + FLAGS.batch_size
			    end_i = end_i + FLAGS.batch_size

		    _, batch_loss_t, batch_logit_t = sess.run([level_opts[i], level_losses[i], level_logits[i]],{level_images_tensor[i]:batch_images, level_labels_tensor[i]:batch_labels })
		    loss_t.append( batch_loss_t)
		    batch_out = np.argmax(batch_logit_t,1)
		    pred_i = 0
		    for pred in batch_out:
			if pred_i>len(level_images[i]):
			    pred_i = 0
			pred_result[i][pred_i] = pred
			pred_i =pred_i + 1
		    #if batch_i == 0:
			#logit_t = np.array(batch_logit_t)
		    #else:
			#logit_t = np.append(logit_t,batch_logit_t,axis=0)
	    #print(np.shape(logit_t))
            level_loss_values[i].append(np.mean( loss_t))
	    level_num_values[i].append(len(level_images[i]))
	    # test accuracy
	    # xxx
            # move samples
            #output = np.argmax(logit_t, 1)
	print('training finished, start to decide which sampes to be moved')
	move_mask = [[[] for k in range(FLAGS.level_number)]for j in range(FLAGS.level_number)]
	nums = [0 for i in range(3)]
	Cnums = [0 for i in range(3)]
	numsDis = [[0 for j in range(10)] for i in range(3)]
	CnumsDis = [[0 for j in range(10)] for i in range(3)]
	predDis = [[0 for j in range(10)] for i in range(3)]
	Acc_log = []
	for i in range(FLAGS.level_number):
	    nums[i]=len(level_images[i])
	    for j in range(nums[i]):
		numsDis[i][level_labels[i][j]] = numsDis[i][level_labels[i][j]] + 1
		predDis[i][pred_result[i][j]] = predDis[i][pred_result[i][j]] + 1
		if pred_result[i][j] == level_labels[i][j]:
		    CnumsDis[i][level_labels[i][j]] = CnumsDis[i][level_labels[i][j]] + 1
		    Cnums[i] = Cnums[i] + 1
		    if np.random.random_sample() < np.exp(-A*T):
			if i>0:
			    move_mask[i][i-1].append(j)
			#if i<1:
			    #move_mask[i][i+1].append(j)
			#elif i>(FLAGS.level_number-2):
			#    move_mask[i][i-1].append(j)
			#else:
			#    if np.random.random_sample() < 0.5:
			#        move_mask[i][i+1].append(j)
			#    else:
			#        move_mask[i][i-1].append(j)
		else:
		    if np.random.random_sample() < np.exp(-B*T):
			if i == 0:
			   move_mask[i][i+1].append(j)
			elif i == FLAGS.level_number-1:
			    if np.random.random_sample()<0.3:
				move_mask[i][i-1].append(j)
			elif np.random.random_sample()<0.3:
			    move_mask[i][i-1].append(j)
			else:
			    move_mask[i][i+1].append(j)
			
			#if i<1:
			#    move_mask[i][i+1].append(j)
			#elif i>(FLAGS.level_number-2):
			#    move_mask[i][i-1].append(j)
			#else:
			#    if np.random.random_sample() < 0.5:
			#        move_mask[i][i+1].append(j)
			#    else:
			#        move_mask[i][i-1].append(j)

	del_mask = [[]for j in range(FLAGS.level_number)]
	iter_acc = []
	Csum=0
	for i in range(FLAGS.level_number):
	    iter_acc.append(Cnums[i]/nums[i])
	    Csum = Csum + Cnums[i]
	iter_acc.append(Csum/50000)
	Acc_log.append(iter_acc)
	print('acc: '+str(Acc_log[-1]))
	for  i in range(FLAGS.level_number):
		print('level'+str(i))
		print(numsDis[i])
		print(CnumsDis[i])
		print(predDis[i])
	print('move samples')
	for i in range(FLAGS.level_number):
	    for k in range(FLAGS.level_number):
	    	print(str(i)+' move to '+str(k)+': '+str(len(move_mask[i][k])))
		level_images[k] = np.append(level_images[k],level_images[i][move_mask[i][k]],axis=0)
	        level_labels[k] = np.append(level_labels[k],level_labels[i][move_mask[i][k]],axis=0)
		del_mask[i] = del_mask[i] + move_mask[i][k]
	    level_images[i] = np.delete(level_images[i],del_mask[i],axis=0)
	    level_labels[i] = np.delete(level_labels[i],del_mask[i],axis=0)

	    
	    print(level_loss_values[i])
	    print(level_num_values[i])



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
