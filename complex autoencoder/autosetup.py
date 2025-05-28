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
tf.keras.saving.get_custom_objects().clear() # remove previously registered objects

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
#@register_keras_serializable
def modrelu(z, b = -0.25):
    """
    Implements modReLU(z) = ReLU(|z| + b) * z/(|z|)
    """
    abs_z = tf.math.abs(z)
    relu = tf.keras.activations.relu(abs_z + b)
    denom = tf.where(abs_z > 0, abs_z, tf.ones_like(abs_z))  # avoid 0 division
    scale = tf.where(abs_z > 0, relu / denom, tf.zeros_like(abs_z))
    return tf.cast(scale, z.dtype) * z

# or....     return tf.maximum(tf.abs(z) + b, 0) * tf.math.exp(1j * tf.math.angle(z))???
    
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

def Jac_modrelu(z, b = -0.25): # todo maybe research sparse implementation, these are possibly going to be large
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

'''Defining custom complex layer object with widely linear transform and custom encoder and decoder'''

#@register_keras_serializable()
class ComplexDense(layers.Layer):
    '''Defines a single layer of the widely linear transform with activation function'''
    def __init__(self, output_dim, activation = modrelu,  name="encoder", **kwargs): 
        super().__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.activation = activation

    # create state of the layer (weight matrices, bias vector)
    def build(self, input_shape):
        self.W1 = self.add_weight(
            shape=(input_shape[-1], self.output_dim),
            initializer=glorot_complex, 
            trainable=True,
            name="W1",
            dtype=tf.complex64,
        )
        self.W2 = self.add_weight(
            shape=(input_shape[-1], self.output_dim),
            initializer=glorot_complex,
            trainable=True,
            name="W2",
            dtype=tf.complex64,
        )
        self.bias = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros", # maybe should check if that works as complex zeros?
            trainable=True,
            name="bias",
            dtype=tf.complex64,
        )

    # helper function computation, useful to do intermediate steps of the forward pass
    def wd_transform(self, inputs):
        # assumes inputs is tf.complex dtype
        inputs = tf.cast(inputs, tf.complex64)
        tf.debugging.assert_type(inputs, tf.complex64) # for debugging
        return tf.matmul(inputs, self.W1) + tf.matmul(tf.math.conj(inputs), self.W2) + self.bias #! left multiplication is std!
    
    # defines the computation
    def call(self, inputs): 
        z = self.wd_transform(inputs)
        return self.activation(z)
    
    # save the model layers
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #          "output_dim": self.output_dim,
    #          "activation": tf.keras.activations.serialize(self.activation)
    #     })
    #     return config
    
    # @classmethod
    # def from_config(cls, config):
    #     config["activation"] = tf.keras.activations.deserialize(config["activation"])
    #     return cls(**config)
       
    

#@register_keras_serializable()
class ComplexEncoder(layers.Layer):
    ''' 
    Maps MNIST digits to compressed input in latent dimension  
    Input/some parameters:
        layer_dims:        int array, e.g. [128, 64, latent_dim]. Does not include input dimension. 
        input_shape:        automatically passed through build, size of original data in the form (batch_size, data_dim)
    
    '''
    def __init__(self, layer_dims, activation=modrelu, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer_dims = layer_dims 
        self.activation = activation
        self.layers_list = []

    def build(self, input_shape):
        dims = [input_shape[-1]] + self.layer_dims # now including the input layer dimension, e.g. [784, 128, 64, latent_dim]
        self.layers_list = [
            ComplexDense(output_dim=dims[i+1], activation=self.activation)
            for i in range(len(self.layer_dims))
        ]
        for layer in self.layers_list:
            layer.build(tf.TensorShape([None, dims[self.layers_list.index(layer)]])) # basically dims[i]: input shape

    def call(self, inputs): # inputs is the data so shape (batch_size, 784)
        x = tf.cast(inputs, tf.complex64)
        for layer in self.layers_list:
            x = layer(x) # this puts x through the layers to the latent dimension
        return x
    
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #          "layer_dims": self.layer_dims,
    #          "activation": tf.keras.activations.serialize(self.activation),
    #          "layers_list": self.layers_list
    #     })
    #     return config
    
    # @classmethod
    # def from_config(cls, config):
    #     config["activation"] = tf.keras.activations.deserialize(config["activation"])
    #     config["layers_list"] = tf.keras.
    #     return cls(**config)

#@register_keras_serializable()
class ComplexDecoder(layers.Layer): 
    ''' 
    Maps input from latent dimension to reconstructed digit 
    Some parameters:
        layer_dims:     int array, e.g. [64, 128, original_dim]. Does not include latent dimension (or input)
    
    '''
    def __init__(self, layer_dims, activation=modrelu, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer_dims = layer_dims 
        self.activation = activation
        self.layers_list = []

    def build(self, input_shape):
        dims = [input_shape[-1]] + self.layer_dims # now including the latent layer dimension, e.g. [latent_dim, 64, 128, original_dim]
        self.layers_list = [
            ComplexDense(output_dim=dims[i+1], activation=self.activation)
            for i in range(len(self.layer_dims))
        ]
        for layer in self.layers_list:
            layer.build(tf.TensorShape([None, dims[self.layers_list.index(layer)]])) # basically dims[i]: input shape

    def call(self, inputs): # inputs is the data so shape (batch_size, 784)
        x = tf.cast(inputs, tf.complex64)
        for layer in self.layers_list:
            x = layer(x) # this puts x through the layers to the latent dimension
        return x