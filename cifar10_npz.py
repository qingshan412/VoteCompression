"""Transfer CIFAR dataset to npz format.
"""
import os, cPickle, argparse
from glob import glob
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("-d","--DataDir", type=str,
                    help="directory for data",
                    default = '../../../data')
parser.add_argument("-m","--Mode", type=str,
                    help="train or test or valid",
                    default = 'train')
parser.add_argument("-s","--DataSet", type=str,
                    help="dataset's name",
                    default = 'cifar10')

args = parser.parse_args()
DataPath = args.DataDir
Mode = args.Mode
DataSet = args.DataSet

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def save2npz(data_path, mode, dataset='cifar10'):
  """Transfer CIFAR image and labels to npz format.

  Args:
    dataset: Either 'cifar10' or 'cifar100'.
    data_path: Filename for data.
    mode: Either 'train' or 'eval'.
  Returns:
    None
  Store:
    images and labels in the format of npz
    features: [100000, image_size, image_size, 3]
    labels: [100000, num_classes]
  Raises:
    ValueError: when the specified dataset is not supported.
  """
  image_size = 32
  channel_size = 3
  
  if dataset != 'cifar10' and dataset != 'cifar100':
    raise ValueError('Not supported dataset %s', dataset)

  if mode == 'train':
    data_files = glob(os.path.join(data_path, dataset, 'data*'))
  else:
    data_files = glob(os.path.join(data_path, dataset, 'test*'))
  
  for i in range(len(data_files)):
    print(data_files[i]+'...')
    data_tmp = unpickle(data_files[i])
    data = data_tmp['data']#10000x3072
    labels_tmp = np.array(data_tmp['labels'])#10000

    m = data.shape[0]
    data = data.reshape((m,channel_size,image_size,image_size))
    features_tmp = np.transpose(data,(0,2,3,1))
    
    if i < 1:
      features = features_tmp
      labels = labels_tmp
    else:
      features = np.vstack((features, features_tmp))
      labels = np.vstack((labels, labels_tmp))
  
  np.savez(os.path.join(data_path, dataset, 'feature_'+ mode +'.npz'), 
    features = features, labels = labels)

if __name__ == "__main__":
  save2npz(data_path=DataPath, mode=Mode, dataset=DataSet)