import argparse
import gzip
import json
import logging
import os
import struct
import boto3

import mxnet as mx
import numpy as np

import pickle
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils import shuffle


def load_pickle_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

def load_data():
    data = load_pickle_file('/opt/ml/code/data.pickle')
    label = load_pickle_file('/opt/ml/code/label.pickle')
    
    return split_data(data, label)

def split_data(data, label):

    
    X, y = (data, label)
    # split dataset
    train_data = X[:80, :].astype('float32')

    train_label = y[:80]
    val_data = X[80 :].astype('float32')
    val_label = y[80:]
    return train_data, train_label, val_data, val_label

def build_graph():
    data = mx.sym.var('data')
    fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, name='act1', act_type="relu")
    fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.sym.Activation(data=fc2, name='act2', act_type="relu")
    fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=5)
    return mx.sym.SoftmaxOutput(data=fc3, name='softmax', multi_output=False)


def get_training_context(num_gpus):
    if num_gpus:
        return [mx.gpu(i) for i in range(num_gpus)]
    else:
        return mx.cpu()

def train(hyperparameters, input_data_config, channel_input_dirs, output_data_dir,
          num_gpus, num_cpus, hosts, current_host, **kwargs):
    train_images, train_labels, val_images, val_labels  = load_data()
    batch_size = 10
    train_iter = mx.io.NDArrayIter(train_images, train_labels, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(val_images, val_labels, batch_size)
    logging.getLogger().setLevel(logging.DEBUG)
    kvstore = 'local' if len(hosts) == 1 else 'dist_sync'
    mlp_model = mx.mod.Module(
        symbol=build_graph(),
        context=get_training_context(num_gpus))
    mlp_model.fit(train_iter,
                  eval_data=val_iter,
                  kvstore=kvstore,
                  optimizer='sgd',
                  optimizer_params={'learning_rate': float(hyperparameters.get("learning_rate", 0.1))},
                  eval_metric='acc',
                  batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                  num_epoch=40)
    return mlp_model

# if __name__ == '__main__':
#     num_gpus = int(os.environ['SM_NUM_GPUS'])
    
#     train(hyperparameters, input_data_config, channel_input_dirs, output_data_dir,
#           num_gpus, num_cpus, hosts, current_host)
