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


def train_network_mnist(autoencoder_architecture, model_file_path):
    tf.reset_default_graph()
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    learning_rate = 0.01
    num_steps = 450
    batch_size = 256
    test_steps = 10 * 4

    input, encoder, decoder = construct_network(autoencoder_architecture)

    target = input
    loss = tf.reduce_mean(tf.pow(target - decoder, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    file_object = open(model_file_path + "log", 'w')

    with tf.Session() as sess:
        sess.run(init)
        for i in range(1, num_steps):
            sample_cnt = 0
            while sample_cnt <= mnist.train.num_examples:
                batch_x, _ = mnist.train.next_batch(batch_size)
                _, loss_value = sess.run([optimizer, loss], feed_dict={input: batch_x})
                sample_cnt += batch_size
            print('Training Iteration %i: Minibatch Loss: %f' % (i, loss_value))
            file_object.write('Training Iteration %i: Minibatch Loss: %f\n' % (i, loss_value))
        saver.save(sess, model_file_path)

        test_cnt = 0
        total_loss_value = 0
        while test_cnt < mnist.test.num_examples:
            batch_x, _ = mnist.train.next_batch(batch_size)
            _, loss_value = sess.run([optimizer, loss], feed_dict={input: batch_x})
            test_cnt += batch_size
            total_loss_value = loss_value + total_loss_value

        sess.close()
        total_loss_value /= test_steps
        print('Test Loss: %f' % (total_loss_value))
        file_object.write('Test Loss: %f\n' % (total_loss_value))
        file_object.close()
        sess.close()

def export_compressed_data(image_data, model_path, autoencoder_architecture, out_path):
    tf.reset_default_graph()
    input, encoder, decoder = construct_network(autoencoder_architecture)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        encoded_image_vector = sess.run(encoder, feed_dict={input: image_data})
        np.savetxt(out_path, encoded_image_vector)
        sess.close()

def import_compressed_data(model_path, autoencoder_architecture,  import_path):
    tf.reset_default_graph()
    input, encoder, decoder = construct_network(autoencoder_architecture)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        recover_encoded_image_vector = np.loadtxt(import_path)
        size = recover_encoded_image_vector.size
        recover_encoded_image_vector = recover_encoded_image_vector.reshape(1, size)
        decoded_image_vector = sess.run(decoder, feed_dict={encoder: recover_encoded_image_vector})
        sess.close()
        return decoded_image_vector


"""
Sample code

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

autoencoder_architecture = [784, 128, 64, 128, 784]

model_path = "C:/Users/saruman/PycharmProjects/Compressive-Autorencoder-for-Image/TestOutput/model_util_test_H_PER.ckpt"
out_path = "C:/Users/saruman/PycharmProjects/Compressive-Autorencoder-for-Image/TestOutput/compressed_data"
batch_x, _ = mnist.train.next_batch(1)


train_network_mnist(autoencoder_architecture , model_path)

export_compressed_data( batch_x , model_path, autoencoder_architecture , out_path)

canvas_orig = np.empty((28, 28))
canvas_orig = batch_x.reshape([28, 28])
print("Original Images")
plt.figure(figsize=(1, 1))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

redata = import_compressed_data(model_path, autoencoder_architecture,  out_path)
canvas_recon = np.empty((28, 28))
canvas_recon = redata.reshape([28, 28])
plt.figure(figsize=(1, 1))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()

"""


