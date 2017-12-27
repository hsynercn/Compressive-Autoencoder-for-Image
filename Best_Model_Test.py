import MyAutoencoder as myAutoencoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
num_steps = 50000
batch_size = 256

average_factor = 10

display_step = 1000
examples_to_show = 10

#autoencoder_architecture_1 = [784, 128, 64, 128, 784]
#autoencoder_architecture_2 = [784, 256, 128, 256, 784]
#autoencoder_architecture_3 = [784, 256, 128, 64, 128, 256, 784]
#autoencoder_architecture_4 = [784, 512, 256, 128, 256, 512, 784]

autoencoder_architecture = [[784, 128, 64, 128, 784],
                            [784, 64, 32, 64, 784],
                            [784, 256, 128, 256, 784],
                            [784, 256, 128, 64, 128, 256, 784],
                            [784, 512, 256, 128, 256, 512, 784]]

#autoencoder_architecture = [[784, 128, 64, 128, 784]]

log_file = open("test_results.txt", "w")

total_loss = []
for i in range(len(autoencoder_architecture)):
    total_loss.append([])
    for j in range(0, num_steps//display_step + 1):
        total_loss[i].append(0)

for model in range(len(autoencoder_architecture)):

    print('model %i: learning rate: %f iteration limit: %i' % (model, learning_rate, num_steps))
    log_file.write('model %i: learning rate:: %f iteration limit: %i\n' % (model, learning_rate, num_steps))
    print(autoencoder_architecture[model])
    log_file.write(str(autoencoder_architecture[model]) + "\n")

    for avarage_counter in range(average_factor):
        print("avarage counter %i" %(avarage_counter))
        input, encoder, decoder = myAutoencoder.construct_network(autoencoder_architecture[model])
        target = input
        loss = tf.reduce_mean(tf.pow(target - decoder, 2))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)
            for i in range(1, num_steps+1):
                batch_x, _ = mnist.train.next_batch(batch_size)
                _, l = sess.run([optimizer, loss], feed_dict={input: batch_x})
                if i % display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f' % (i, l))
                    total_loss[model][i//display_step] = l + total_loss[model][i//display_step]



    for k in range(num_steps//display_step):
        total_loss[model][k] = total_loss[model][k] / average_factor
        print('Step %i: Avarage Minibatch Loss: %f' % (k, total_loss[model][k]))
        log_file.write('Step %i: Avarage Minibatch Loss: %f\n' % (k, total_loss[model][k]))

log_file.close()