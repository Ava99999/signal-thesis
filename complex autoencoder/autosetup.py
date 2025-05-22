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

'''Contains the classes and initialisatinons of the autoencoder model'''

''' Initializers, loss- and activation functions '''
# complex initianalizer
def glorot_complex(shape, dtype=tf.complex64):
    real = (1/np.sqrt(2))*tf.keras.initializers.GlorotUniform()(shape, dtype=tf.float32)
    imag = (1/np.sqrt(2))*tf.keras.initializers.GlorotUniform()(shape, dtype=tf.float32) #tf.keras.initializers.RandomNormal(stddev=1e-4)(shape, dtype=tf.float32)
    return tf.complex(real, imag)

# custom activation function 
def arctan_complex(z):
    i = tf.constant(1j, dtype=tf.complex64)
    return 0.5*i*(tf.math.log(1 - i*z) - tf.math.log(1 + i*z))

# modReLU activation function
def modrelu(z, b = -1.):
    """
    Implements modReLU(z) = ReLU(|z| + b) * z/(|z|)
    """
    abs_z = tf.math.abs(z)
    relu = tf.keras.activations.relu(abs_z + b)
    denom = tf.where(abs_z > 0, abs_z, tf.ones_like(abs_z))  # avoid 0 division
    scale = tf.where(abs_z > 0, relu / denom, tf.zeros_like(abs_z))
    return tf.cast(scale, z.dtype) * z
    
#MSE loss function (single sample)
def loss_MSE(a,x):
    '''
    Compute MSE loss function with output final layer (a) and input (x). Both assumed Tensor Complex64 1D arrays.
    Formula: (a - x)^H (a - x) = sum_j (Re[a_j - x_j])^2 + (Im[a_j - x_j])^2
    '''
    subtract = a-x
    real_part = tf.square(tf.math.real(subtract))
    imag_part = tf.square(tf.math.imag(subtract))
    return tf.reduce_sum(real_part + imag_part)


''' Supporting derivatives to be used in the backpropagation algorithm '''
def dLossdaL(a, x):
    '''
    Wirtinger derivatives of derivative of the MSE loss function. Requires that a and x have the same size (expected 1D complex arrays as a row vector).
    
    Input:
        a:      Tensor complex64 array, represents final activation
        x:      Tensor complex64 array, represents input sample

    Output: R and R*-derivative of loss function.
    '''
    return tf.math.conj(a - x), a - x

def Jac_modrelu(z, b = -1.): # todo maybe research sparse implementation, these are possibly going to be large
    '''
    Jacobian of the modrelu function with radius b. 
    
    Input:
        z:      Tensor complex64 array
        b:      float, 'dead zone' parameter, assumed to be negative
    '''
    abs_z     = tf.cast(tf.math.abs(z), z.dtype) # has to remain complex64 for operations
    dev_z     = tf.where(tf.math.abs(z) + b >= 0, tf.ones_like(z) + tf.cast((b/2)/abs_z, z.dtype), tf.zeros_like(z)) 
    dev_zstar = tf.where(tf.math.abs(z) + b >= 0, tf.cast((-b*z**2)/(2*abs_z**3), z.dtype), tf.zeros_like(z))
    J_z       = tf.linalg.diag(dev_z)
    J_zstar   = tf.linalg.diag(dev_zstar)
    return J_z, J_zstar



@register_keras_serializable()
class ComplexEncoder(layers.Layer):
    ''' Maps MNIST digits to compressed input in latent dimension  '''
    def __init__(self, latent_dim, activation = arctan_complex,  name="encoder", **kwargs): 
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.activation = activation
    
    # create state of the layer (weight matrices, bias vector)
    def build(self, input_shape):
        self.W1 = self.add_weight(
            shape=(input_shape[-1], self.latent_dim),
            initializer=glorot_complex, 
            trainable=True,
            name="W1",
            dtype=tf.complex64,
        )
        self.W2 = self.add_weight(
            shape=(input_shape[-1], self.latent_dim),
            initializer=glorot_complex,
            trainable=True,
            name="W2",
            dtype=tf.complex64,
        )
        self.bias = self.add_weight(
            shape=(self.latent_dim,),
            initializer="zeros", # maybe should check if that works as complex zeros?
            trainable=True,
            name="bias",
            dtype=tf.complex64,
        )
    
    # defines the computation
    def call(self, inputs): 
        # assumes inputs is tf.complex dtype
        inputs = tf.cast(inputs, tf.complex64)
        tf.debugging.assert_type(inputs, tf.complex64) # for debugging
        z = tf.matmul(inputs, self.W1) + tf.matmul(tf.math.conj(inputs), self.W2) + self.bias
        return self.activation(z)
    
@register_keras_serializable()
class ComplexDencoder(layers.Layer):
    ''' Maps input from latent dimension to reconstructed digit  '''
    def __init__(self, original_dim, activation = arctan_complex, name="decoder", **kwargs): 
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.activation = activation

    # create state of the layer (weight matrices, bias vector)
    def build(self, input_shape):
        self.W1 = self.add_weight(
            shape=(input_shape[-1], self.original_dim),
            initializer=glorot_complex, # default option: can also do random_normal, no idea what would be good
            trainable=True,
            name="W1",
            dtype=tf.complex64,
        )
        self.W2 = self.add_weight(
            shape=(input_shape[-1], self.original_dim),
            initializer=glorot_complex, # default option: can also do random_normal, no idea what would be good
            trainable=True,
            name="W2",
            dtype=tf.complex64,
        )
        self.bias = self.add_weight(
            shape=(self.original_dim,),
            initializer="zeros",
            trainable=True,
            name="bias",
            dtype=tf.complex64,
        )
    
    # defines the computation
    def call(self, inputs): 
        # assumes inputs is tf.complex64 dtype
        inputs = tf.cast(inputs, tf.complex64)
        tf.debugging.assert_type(inputs, tf.complex64) # for debugging
        z = tf.matmul(inputs, self.W1) + tf.matmul(tf.math.conj(inputs), self.W2) + self.bias
        return self.activation(z)
