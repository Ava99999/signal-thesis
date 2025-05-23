import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import autosetup

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable, deserialize_keras_object
from tensorflow.test import compute_gradient

from scipy.sparse.linalg import LinearOperator
from scipy.fft import fft, ifft, fft2, ifft2
from scipy.optimize import minimize

'''Contains the backpropagation algorithm and gradient logic'''
# TODO many things are made for a single sample and not for batch logic!

''' Helper functions that are repeated in CBP '''
def compute_grad_W1(a_prev, dR_dq_star): 
    a_prev_star = tf.conj(a_prev)
    return tf.einsum('i,j->ij', dR_dq_star, a_prev_star) #(uv^T) output[i,j] = u[i]*v[j]

def compute_grad_W2(a_prev, dR_dq_star):
    return tf.einsum('i,j->ij', dR_dq_star, a_prev)

#@tf.function # add at the end once stable; check/avoid numpy operations
def CBP(x, y, encoder, decoder, loss_fn, dev_loss, jac_act): 
    '''
    we are going to write complex backpropagation in a stupid way as if it was an ordinary function and see what we need, 
    and then later maybe break it up into more sophisticated pieces

    Input:
        x:          Tensor complex64, specific training sample (might change to a batch of samples later)
        y:          Tensor complex64, result of forward pass, final output. Should equal aL
        encoder:    Keras layer object (function), contains weights and forward pass operations
        decoder:    Keras layer object (function), "
        loss_fn:    Loss function, maps from complex tensor to real Tensor constant
        dev_loss:   Derivatives of the loss function, should return two arguments
        jac_act:    Jacobian function of the activation function (C -> C)

    Output:
        grads:      List of gradients with respect to all weights
    '''
    ### FORWARD PASS ###
    # NOTE: the order of this list is l = 1, 2, ..., L of the layers from encoder to decoder
    a = x # first activation: a0
    q_list, a_list = [], []

    # record activations encoder
    for layer in encoder.layers_list:
        q = layer.wd_transform(a)
        a = layer.activation(q)
        q_list.append(q)
        a_list.append(a)
    
    # record activations decoder
    for layer in decoder.layers_list:
        q = layer.wd_transform(a)
        a = layer.activation(q)
        q_list.append(q)
        a_list.append(a) 

    # TODO maybe compute the value of the loss function? or do that in model or something? maybe not necessary here

    # TODO test if last element of a_list is equal to y

    ### BACKWARD PASS ###
    # NOTE: the order of this list is l = L, L-1, ..., 1 of the layers from decoder to encoder
    dR_dq_list, dR_dqstar_list  = [], [] # record derivatives of the loss function up to pre-activation q^l, both R- and R*-derivative
    grads = [] # save gradients of all parameters in the format [(grad_W1[1], layer[1].W1), ..., (grad_b[L], layer[L].b)]

    # Wirtinger derivatives final layer
    dR_daL, dR_daL_star   = dev_loss(y,x) # derivatives of loss function wrt final layer output
    daL_dqL, daL_dqL_star = jac_act(q_list[-1]) # derivatives of activation function wrt final widely lin. transformation
    dR_dqL                = dR_daL @ daL_dqL        + dR_daL_star @ tf.math.conj(daL_dqL_star) # R-derivative
    dR_dqL_star           = dR_daL @ daL_dqL_star   + dR_daL_star @ tf.math.conj(daL_dqL) # R*-derivative

    dR_dq_list.append(dR_dqL)
    dR_dqstar_list.append(dR_dqL_star)

    # compute gradients final layer
    grad_W1_L = compute_grad_W1(a_list[-2], dR_dqL_star) # a[L-1]* x dR/dqL*
    grad_W2_L = compute_grad_W2(a_list[-2], dR_dqL_star) # a[L-1]  x dR/dqL*
    grad_b_L  = dR_dqL_star 
    layer_L   = decoder.layers_list[-1]

    grads.append((grad_W1_L, layer_L.W1))
    grads.append((grad_W2_L, layer_L.W2))
    grads.append((grad_W2_L, layer_L.bias))

    # TODO now recursively

    return grads