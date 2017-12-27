import MyAutoencoder as myAutoencoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
num_steps = 50000
batch_size = 256
average_factor = 1

display_step = 1000

autoencoder_architecture = [784, 128, 64, 128, 784]
log_file = open("compression_results.txt", "w")


input, encoder, decoder = myAutoencoder.construct_network(autoencoder_architecture)

target = input
loss = tf.reduce_mean(tf.pow(target - decoder, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

#with tf.Session() as sess:#
#    sess.run(init)
#    for i in range(1, num_steps+1):
#        batch_x, _ = mnist.train.next_batch(batch_size)
#        _, l = sess.run([optimizer, loss], feed_dict={input: batch_x})
#        if i % display_step == 0 or i == 1:
#            print('Step %i: Minibatch Loss: %f' % (i, l))
#            log_file.write('Step %i: Minibatch Loss: %f\n' % (i, l))
#    saver.save(sess, 'C:/Users/saruman/PycharmProjects/compression_model.ckpt')

n = 1
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))



with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "C:/Users/saruman/PycharmProjects/compression_model.ckpt")
    print("Model restored.")

    # MNIST test set
    batch_x, _ = mnist.test.next_batch(n)
    # Encode and decode the digit image
    g = sess.run(decoder, feed_dict={input: batch_x})

    encoded_image_vector = sess.run(encoder, feed_dict={input: batch_x})

    np.savetxt("original_data.txt", batch_x)

    np.savetxt("compressed_data.txt", encoded_image_vector)

    recovere_encoded_image_vector = np.loadtxt("compressed_data.txt")

    size = recovere_encoded_image_vector.size
    recovere_encoded_image_vector = recovere_encoded_image_vector.reshape(1, size)

    decoded_image_vector = sess.run(decoder, feed_dict={encoder: recovere_encoded_image_vector})

    canvas_orig = batch_x.reshape([28, 28])
    canvas_recon = decoded_image_vector.reshape([28, 28])


print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()
log_file.close()