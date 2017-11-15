# coding: utf-8
import time
import sys

import IPython
from IPython.core.display import display, HTML
from IPython.display import SVG

#############################################
import matplotlib
from matplotlib import pyplot as plt

import tabulate
from tabulate import tabulate

##################################
import numpy as np
from numpy.linalg import inv, svd
from numpy import mean, std

import pandas as pd

import pywt#Walvet coeff

import scipy 
from scipy import optimize
from scipy.signal import filtfilt, butter, lfilter, cwt, morlet, ricker
from scipy.stats import norm


######## Sclearn
import sklearn
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, log_loss,accuracy_score

if sys.version_info >= (3,5):
    
    ######## Tensorflow
    import tensorflow as tf
    from tensorflow.python.client import device_lib


    ######## Keras
    import keras
    from keras import backend as K
    from keras import losses
    from keras import regularizers
    from keras.callbacks import Callback, ModelCheckpoint, History,EarlyStopping
    from keras.engine.topology import Layer
    from keras.models import Sequential, Model
    from keras.optimizers import Adam
    from keras.optimizers import SGD
    from keras.layers import Input
    from keras.layers.core  import Dense, Activation, Dropout, Flatten, Reshape
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv1D,Conv2D
    from keras.layers.pooling import MaxPooling1D
    from keras.layers.recurrent import SimpleRNN, GRU, LSTM
    from keras.layers.local import LocallyConnected1D, LocallyConnected2D
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.utils.visualize_util import model_to_dot
    from keras.utils import plot_model




#####################################################
#display(HTML("<style>.container { width:70% !important; }</style>"))
print (sys.version)
if sys.version_info >= (3,5):
    print ('TF version:',tf.__version__)
    print ("Keras version:",keras.__version__)
    print (" ")
    print ("##")
    print (device_lib.list_local_devices())
    print ("##")