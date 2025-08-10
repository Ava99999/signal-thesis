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
from tensorflow.test import compute_gradient

from scipy.sparse.linalg import LinearOperator
from scipy.fft import fft, ifft, fft2, ifft2
from scipy.optimize import minimize

'''This file contains the backpropagation algorithm and gradient logic'''

''' Helper functions that are repeated in CBP '''
def compute_grad_W1(a_prev, dR_dq_star): 
    a_prev_star = tf.math.conj(a_prev)
    return tf.einsum('i,j->ij', tf.squeeze(dR_dq_star), tf.squeeze(a_prev_star)) #(uv^T) output[i,j] = u[i]*v[j]

def compute_grad_W2(a_prev, dR_dq_star):
    return tf.einsum('i,j->ij', tf.squeeze(dR_dq_star), tf.squeeze(a_prev)) #tf.squeeze removes the dimensions of length 1

def CBP(x, y, encoder, decoder, dev_loss, jac_act): 
    '''
    Complex backpropagation algorithm for the widely linear transform, used for training the complex autoencoder.

    Input:
        x:          Tensor complex64, specific training sample
        y:          Tensor complex64, result of forward pass, final output. Should be equal to aL
        encoder:    Keras layer object (function), contains weights and forward pass operations
        decoder:    Keras layer object (function), contains weights and forward pass operations
        dev_loss:   Derivatives of the loss function, should return two arguments
        jac_act:    Jacobian function of the activation function (C -> C)

    Output:
        grads:      List of gradients with respect to all weights
    '''
    ### FORWARD PASS ###
    # NOTE: the order of this list is l = 1, 2, ..., L of the layers from encoder to decoder
    # NOTE: q_list contains q1, q2, ..., qL, a_list contains a0, ..., aL
    a = x # first activation: a0
    q_list, a_list = [], []
    a_list.append(x) 

    # record activations encoder & decoder
    full_layers = encoder.layers_list + decoder.layers_list
    L = len(full_layers)
    for layer in full_layers:
        q = layer.wd_transform(a)
        a = layer.activation(q)
        q_list.append(q)
        a_list.append(a)

    # test if last element of a_list is equal to y
    tf.debugging.assert_equal(y, a_list[-1], message="a^L does not match y")

    ### BACKWARD PASS ###
    # NOTE: the order of this list is l = L, L-1, ..., 1 of the layers from decoder to encoder
    dR_dq_list, dR_dqstar_list  = [], [] # record derivatives of the loss function up to pre-activation q^l, both R- and R*-derivative
    grads = [] # save gradients of all parameters in the format [(grad_W1[1], layer[1].W1), ..., (grad_b[L], layer[L].b)]

    # Wirtinger derivatives final layer
    dR_daL, dR_daL_star   = dev_loss(y,x) # derivatives of loss function wrt final layer output
    daL_dqL, daL_dqL_star = jac_act(q_list[-1]) # derivatives of activation function wrt final widely lin. transformation
    dR_dqL                = dR_daL @ daL_dqL        + dR_daL_star @ tf.math.conj(daL_dqL_star) # R-derivative
    dR_dqL_star           = dR_daL @ daL_dqL_star   + dR_daL_star @ tf.math.conj(daL_dqL)      # R*-derivative

    dR_dq_list.append(dR_dqL)
    dR_dqstar_list.append(dR_dqL_star)

    # compute gradients final layer
    grad_W1_L = 2*compute_grad_W1(a_list[L-1], dR_dqL_star) # a[L-1]* x dR/dqL*
    grad_W2_L = 2*compute_grad_W2(a_list[L-1], dR_dqL_star) # a[L-1]  x dR/dqL*
    grad_b_L  = 2*tf.squeeze(dR_dqL_star)                   # remove dimensions of size 1
    layer_L   = full_layers[-1]

    grads.append((tf.transpose(grad_W1_L), layer_L.W1))
    grads.append((tf.transpose(grad_W2_L), layer_L.W2))
    grads.append((tf.transpose(grad_b_L), layer_L.bias))

    # recursive compute gradients layers L-1, ..., 1
    for l in reversed(range(1, L)):
        layer_l                         = full_layers[l]
        layer_lprev                     = full_layers[l-1] 
        dal_dql_prev, dal_dql_prev_star = jac_act(q_list[l-1])     # da[l-1]/dq[l-1]
        dql_dal_prev                    = tf.transpose(layer_l.W1) # dq[l]/da[l-1]
        dql_dal_prev_star               = tf.transpose(layer_l.W2)

        # compute dR/dq[l-1] and dR/dq[l-1]*
        beta  = dql_dal_prev @ dal_dql_prev      + dql_dal_prev_star @ tf.math.conj(dal_dql_prev_star) # R-derivative
        gamma = dql_dal_prev @ dal_dql_prev_star + dql_dal_prev_star @ tf.math.conj(dal_dql_prev)      # R*-derivative

        dR_dql_prev      = (dR_dq_list[L-(l+1)])[-1] @ beta  + (dR_dqstar_list[L-(l+1)])[-1] @ tf.math.conj(gamma) # R-derivative
        dR_dql_prev_star = (dR_dq_list[L-(l+1)])[-1] @ gamma + (dR_dqstar_list[L-(l+1)])[-1] @ tf.math.conj(beta)  # R*-derivative

        dR_dq_list.append(dR_dql_prev)
        dR_dqstar_list.append(dR_dql_prev_star)

        # compute gradients wrt W1[l-1], W2[l-1], b[l-1]
        grad_W1_lprev = 2*compute_grad_W1(a_list[l-1], dR_dql_prev_star) # a[l-1]* \cdot dR/dq[l-1]*
        grad_W2_lprev = 2*compute_grad_W2(a_list[l-1], dR_dql_prev_star) # a[l-1]  \cdot dR/dq[l-1]*
        grad_b_lprev  = 2*tf.squeeze(dR_dql_prev_star)                   # remove dimensions of size 1

        grads.append((tf.transpose(grad_W1_lprev), layer_lprev.W1))
        grads.append((tf.transpose(grad_W2_lprev), layer_lprev.W2))
        grads.append((tf.transpose(grad_b_lprev), layer_lprev.bias))

    return grads


def CBP_decoder(z, decoder, jac_act): 
    '''
    Complex backpropagation algorithm for the widely linear transform of only the decoder layers. Used to compute the Jacobian of the prior.

    Input:
        z:          Tensor complex64, latent space representation, required to have dimension of latent_space
        decoder:    Keras layer object (function), contains weights and forward pass operations
        jac_act:    Jacobian function of the activation function (C -> C)

    Output:
        jacobians:  the 2 Jacobians of the decoder with respect to the latent input
    '''
    ### FORWARD PASS ###
    # NOTE: the order of this list is l = l_d, ..., L of the layers from encoder to decoder
    # NOTE: q_list contains ql_d, ql_d+1, ..., qL, a_list contains al_d-1, ..., aL
    a = z # first activation: al_[d]-1
    q_list, a_list = [], []
    a_list.append(z) 

    # record activations decoder
    D = len(decoder.layers_list)
    for layer in decoder.layers_list:
        q = layer.wd_transform(a)
        a = layer.activation(q)
        q_list.append(q)
        a_list.append(a)

    ### BACKWARD PASS ###
    # NOTE: the order of this list is l = L, L-1, ..., l_D of the layers from decoder
    da_dq_list, da_dqstar_list  = [], [] # record derivatives of the loss function up to pre-activation q^l, both R- and R*-derivative
 
    # Wirtinger derivatives final layer
    daL_dqL, daL_dqL_star = jac_act(q_list[-1]) # derivatives of activation function wrt final widely lin. transformation

    da_dq_list.append(daL_dqL)
    da_dqstar_list.append(daL_dqL_star)

    # recursive compute gradients layers L-1, ..., l_d+1
    for l in reversed(range(1, D)):
        layer_l                         = decoder.layers_list[l]
        layer_lprev                     = decoder.layers_list[l-1] 
        dql_dal_prev                    = tf.transpose(layer_l.W1) # dq[l]/da[l-1]
        dql_dal_prev_star               = tf.transpose(layer_l.W2)
        dal_dql_prev, dal_dql_prev_star = jac_act(q_list[l-1])     # da[l-1]/dq[l-1]

        # compute da/dq[l-1] and da/dq[l-1]*
        beta  = dql_dal_prev @ dal_dql_prev      + dql_dal_prev_star @ tf.math.conj(dal_dql_prev_star) # R-derivative
        gamma = dql_dal_prev @ dal_dql_prev_star + dql_dal_prev_star @ tf.math.conj(dal_dql_prev)      # R*-derivative

        # TODO size check: it might not be the [-1], D-(l+1) is correct though
        da_dql_prev      = (da_dq_list[D-(l+1)])[-1] @ beta  + (da_dqstar_list[D-(l+1)])[-1] @ tf.math.conj(gamma) # R-derivative
        da_dql_prev_star = (da_dq_list[D-(l+1)])[-1] @ gamma + (da_dqstar_list[D-(l+1)])[-1] @ tf.math.conj(beta)  # R*-derivative

        da_dq_list.append(da_dql_prev)
        da_dqstar_list.append(da_dql_prev_star)
    
    daL_dz      = (da_dq_list[-1])[-1] @ tf.transpose(layer_lprev.W1) + (da_dqstar_list[-1])[-1] @ tf.math.conj(tf.transpose(layer_lprev.W2))
    daL_dzstar  = (da_dq_list[-1])[-1] @ tf.transpose(layer_lprev.W2) + (da_dqstar_list[-1])[-1] @ tf.math.conj(tf.transpose(layer_lprev.W1))

    return daL_dz, daL_dzstar

def CBP_decoder_v2(z, decoder, jac_act): 
    '''
    Complex backpropagation algorithm for the widely linear transform of only the decoder layers. Used to compute the Jacobian of the prior.
    As a debugging check, this computes the same Jacobians as CBP_decoder, but using a different breakdown of the chain rule.

    Input:
        z:          Tensor complex64, latent space representation, required to have dimension of latent_space
        encoder:    Keras layer object (function), contains weights and forward pass operations
        decoder:    Keras layer object (function), "
        loss_fn:    todo
        dev_loss:   Derivatives of the loss function, should return two arguments
        jac_act:    Jacobian function of the activation function (C -> C)

    Output:
        jacobians:  the 2 Jacobians of the decoder with respect to the latent input
    '''
    ### FORWARD PASS ###
    # NOTE: the order of this list is l = l_d, ..., L of the layers from encoder to decoder
    # NOTE: q_list contains ql_d, ql_d+1, ..., qL, a_list contains al_d-1, ..., aL
    a = z # first activation: al_[d]-1
    q_list, a_list = [], []
    a_list.append(z) 

    # record activations decoder
    D = len(decoder.layers_list)
    layer_L = decoder.layers_list[-1]
    for layer in decoder.layers_list:
        q = layer.wd_transform(a)
        a = layer.activation(q)
        q_list.append(q)
        a_list.append(a)

    ### BACKWARD PASS ###
    # NOTE: the order of this list is l = L, L-1, ..., l_D of the layers from decoder
    dG_da_list, dG_dastar_list  = [], [] # record derivatives of the decoder function up to activation a^l, both R- and R*-derivative
 
    # Wirtinger derivatives final layer: jac_act(q^L), has shape (1, dim, dim)
    daL_dqL, daL_dqL_star = jac_act(q_list[-1]) # derivatives of activation function wrt final widely lin. transformation

    # dG/da[L-1] = da[L]/da[L-1]
    dG_daL_prev      = daL_dqL[0] @ tf.transpose(layer_L.W1) + daL_dqL_star[0] @ tf.math.conj(tf.transpose(layer_L.W2))
    dG_daL_prev_star = daL_dqL[0] @ tf.transpose(layer_L.W2) + daL_dqL_star[0] @ tf.math.conj(tf.transpose(layer_L.W1))

    dG_da_list.append(dG_daL_prev)
    dG_dastar_list.append(dG_daL_prev_star)

    # recursive compute gradients layers L-1, ..., l_d+1
    for l in reversed(range(1, D)):
        layer_l                         = decoder.layers_list[l]       # starts at final layer D-1
        layer_lprev                     = decoder.layers_list[l-1] 
        dql_dal_prev                    = tf.transpose(layer_lprev.W1) # dq[l]/da[l-1]
        dql_dal_prev_star               = tf.transpose(layer_lprev.W2)
        dal_dql_prev, dal_dql_prev_star = jac_act(q_list[l-1])         # da[l-1]/dq[l-1]

        # compute da[l]/da[l-1] and da[l]/da[l-1]*
        rho = dal_dql_prev[0] @ dql_dal_prev      + dal_dql_prev_star[0] @ tf.math.conj(dql_dal_prev_star) # R-derivative
        tau = dal_dql_prev[0] @ dql_dal_prev_star + dal_dql_prev_star[0] @ tf.math.conj(dql_dal_prev)      # R*-derivative

        dG_dal_prev      = (dG_da_list[D-(l+1)]) @ rho  + (dG_dastar_list[D-(l+1)]) @ tf.math.conj(tau) # R-derivative
        dG_dal_prev_star = (dG_da_list[D-(l+1)]) @ tau + (dG_dastar_list[D-(l+1)]) @ tf.math.conj(rho)  # R*-derivative

        dG_da_list.append(dG_dal_prev)
        dG_dastar_list.append(dG_dal_prev_star)
    
    dG_dz      = dG_da_list[-1]     
    dG_dzstar  = dG_dastar_list[-1]
    
    return dG_dz, dG_dzstar