import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt




def construct_network(layer_neurons):


    weights = list()
    biases = list()
    for i in range(len(layer_neurons) - 1):
        weights.append(tf.Variable(tf.random_normal([layer_neurons[i], layer_neurons[i + 1]])))

    for i in range(len(layer_neurons)):
        biases.append(tf.Variable(tf.random_normal([layer_neurons[i]])))

    layers = list()
    layers.append(tf.placeholder("float", [None, layer_neurons[0]]))

    for i in range(1, len(layer_neurons)):
        layers.append(tf.nn.sigmoid(tf.add(tf.matmul(layers[i - 1], weights[i -1 ]), biases[i])))

    return layers[0], layers[(len(layers) - 1)//2], layers[len(layers) - 1]
