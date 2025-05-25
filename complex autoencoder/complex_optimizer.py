import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable, deserialize_keras_object
from tensorflow.keras.optimizers import Optimizer
from tensorflow.test import compute_gradient
from tensorflow.compat.v1 import assign_sub

from scipy.sparse.linalg import LinearOperator
from scipy.fft import fft, ifft, fft2, ifft2
from scipy.optimize import minimize

'''Contains the SGD coptimizer and step-size computation'''
def adaptive_stepsize(grads_and_vars):
    ''' Currently just use a scalar, has to return a complex64 tensor ''' 
    return tf.constant(1e-4, dtype=tf.complex64)

class Complex_SGD(Optimizer):
    '''Subclass op keras Optimizer, to be used in the keras model'''
    def __init__(self, stepsize_fn, learning_rate=1e-4, name="ComplexSGD", **kwargs): # later plug std for stepsize_fn
        super().__init__(learning_rate = learning_rate, name=name, **kwargs)
        self.stepsize_fn = stepsize_fn

    def apply_gradients(self, grads_and_vars, **kwargs):
        learning_rate = self.stepsize_fn(grads_and_vars) # note this does not exist yet
        for grad, var in grads_and_vars:
            var.assign_sub(learning_rate * grad)
        return None