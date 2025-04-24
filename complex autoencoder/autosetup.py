import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import random
#import pylops # might not need
import math
import pyproximal

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

# complex initianalizer
def glorot_complex(shape, dtype=tf.complex64):
    real = tf.keras.initializers.GlorotUniform()(shape, dtype=tf.float32)
    imag = tf.keras.initializers.RandomNormal(stddev=1e-4)(shape, dtype=tf.float32)
    return tf.complex(real, imag)

# custom activation function 
def arctan_complex(z):
    i = tf.constant(1j, dtype=tf.complex64)
    return 0.5*i*(tf.math.log(1 - i*z) - tf.math.log(1 + i*z))

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
            initializer=glorot_complex, # default option: can also do random_normal, no idea what would be good
            trainable=True,
            name="W1",
            dtype=tf.complex64,
        )
        self.W2 = self.add_weight(
            shape=(input_shape[-1], self.latent_dim),
            initializer=glorot_complex, # default option: can also do random_normal, no idea what would be good
            trainable=True,
            name="W2",
            dtype=tf.complex64,
        )
        self.bias = self.add_weight(
            shape=(self.latent_dim,),
            initializer="zeros",
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
