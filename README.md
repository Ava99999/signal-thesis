## Code for thesis: Using a complex autoencoder as generative prior in phase retrieval

In the folders, the code for my thesis submitted in partial fulfillment of my MSc degree at Utrecht University can be found. 
The datasets used for training the model are the MNIST dataset of handwritten digits (Lecun, Cortes and Burges [2010](http://yann.lecun.com/exdb/mnist)) and Fashion MNIST (Xiao, Rasul and Vollgraf, [2017](https://arxiv.org/abs/1708.07747)).
The architecture of the code for phase retrieval, which encompasses the minimization using quasi Newton L-BFGS, was retrieved from Aslan et al., [2025](https://arxiv.org/pdf/2502.01338) and can be found in this [Github repository](https://github.com/TristanvanLeeuwen/PtyGenography). 
The code has been written assisted by OpenAI's CoPilot (OpenAI, [2025](chatgpt.com)).

The code uses TensorFlow with the Keras API, and is based on the TensorFlow tutorials (Chollet, [2015a](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch), [2015b](https://www.tensorflow.org/tutorials/generative/autoencoder)).
