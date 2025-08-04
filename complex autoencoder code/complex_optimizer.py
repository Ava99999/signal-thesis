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
# NOTE does not contain a failsafe for gradient blow-up or vanishing of alpha

def adaptive_stepsize(x, y, alpha, encoder, decoder, loss_fn, grads_and_vars, max_trials=5):
    '''
    Backtracking line search to find a step size satisfying loss_fn(y_new) <= loss_fn(y) - alpha*c1*|grad(loss_fn(y))|^2, where grad is wrt the weights.
    
    Input:
        x:              Tensor complex64, specific training sample
        y:              Tensor complex64, result of forward pass
        alpha:          float, initial step size    
        encoder:        Keras layer object (function), contains weights and forward pass operations
        decoder:        Keras layer object (function), contains weights and forward pass operations
        loss_fn:        loss function (C^M -> R) with M the signal dimension
        grads_and_vars: list of gradients wrt weight matrices and bias vectors and respective variables
        max_trials:     int, maximum attempts of the line search

    Output:
        alpha_final:    float, final step size
    ''' 

    c1 = tf.constant(1e-4, dtype=tf.float32)
    rho = tf.constant(0.5, dtype=tf.float32)

    grad_norm_squared = tf.add_n([tf.reduce_sum(tf.abs(g)**2) for g, _ in grads_and_vars])
    current_loss = loss_fn(y,x)

    snapshots = [tf.identity(v) for _, v in grads_and_vars] 

    def cond(trial, alpha, success):
        return tf.logical_and(trial < max_trials, tf.logical_not(success))

    def body(trial, alpha, success):
        alpha = alpha * tf.pow(rho, tf.cast(trial, tf.float32))

        # simulate step
        for grad, var in grads_and_vars:
            var.assign_sub(tf.cast(alpha, tf.complex64) * grad)

        y_update = decoder(encoder(x))
        new_loss = loss_fn(y_update, x)
        compare_loss = current_loss - c1*alpha*grad_norm_squared

        # undo step
        for (grad, var), snapshot in zip(grads_and_vars, snapshots):
            var.assign(snapshot)

        # check armijo condition
        success = new_loss <= compare_loss
        return trial + 1, alpha, success

    # initialize loop vars
    trial = tf.constant(0)
    success = tf.constant(False)

    trial, alpha_final, _ = tf.while_loop(
        cond, body, [trial, alpha, success]
    )

    return alpha_final

class Complex_SGD(Optimizer):
    '''Subclass op keras Optimizer, to be used in the keras model'''
    def __init__(self, name="ComplexSGD", **kwargs): 
        super().__init__(learning_rate = 1e-2, name=name, **kwargs)

    def apply_gradients(self, grads_and_vars, alpha = None, **kwargs):
        if alpha is None:
            alpha = 1e-2
        for grad, var in grads_and_vars:
            var.assign_sub(tf.cast(alpha, tf.complex64) * grad)
        
        return None