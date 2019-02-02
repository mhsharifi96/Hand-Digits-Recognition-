import tensorflow as tf
import numpy as np
# np.set_printoptions(threshold=np.nan)

from pprint import pprint
from random import sample
from matplotlib import pyplot as plt
from HodaDatasetReader import read_hoda_dataset

from PIL import Image
def reader():
    
    
    print('Reading train dataset (Train 60000.cdb)...')
    X_train, Y_train = read_hoda_dataset(dataset_path='../DigitDB/Train 60000.cdb',
                                images_height=32,
                                images_width=32,
                                one_hot=True,  
                                reshape=True)
    print('Reading test dataset (Test 20000.cdb)...')
    X_test, Y_test = read_hoda_dataset(dataset_path='../DigitDB/Test 20000.cdb',
                                images_height=32,
                                images_width=32,
                                one_hot=True,
                                reshape=True)
    print('Reading remaining samples dataset (RemainingSamples.cdb)...')
    X_remaining, Y_remaining = read_hoda_dataset('../DigitDB/RemainingSamples.cdb',
                                                images_height=32,
                                                images_width=32,
                                                one_hot=True,
                                                reshape=True)
   
    return X_train, Y_train, X_test, Y_test, X_remaining, Y_remaining

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, X_remaining, Y_remaining = reader()
    
    n_train = 60000
    n_remaining = 22352
    n_test = 20000

    n_input = 1024   # input layer (32x32 pixels)
    n_hidden1 = 50  # 1st hidden layer
    n_hidden2 = 20  # 2nd hidden layer
    n_output = 10

    learning_rate = 1e-4
    n_iterations = 1000

    dropout = 0.5
    

    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])
    keep_prob = tf.placeholder(tf.float32) 

    weights = {
        'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
        'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden2, n_output], stddev=0.1)),
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
    }

    layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_drop = tf.nn.dropout(layer_2, keep_prob)
    output_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output_layer))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    
    print()
    print('Training...')

    chunc_size = 100

    for i in range(int(n_train/chunc_size)):
        batch_x = []
        batch_y = []
        for j in range(i*chunc_size, (i+1)*chunc_size-1):
            batch_x.append(X_train[j])
            batch_y.append(Y_train[j])

        sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

        if i%10==0:
            minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
            print("Iteration", str(i*chunc_size), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))

    

    print('Testing...')

    test_accuracy = 0
    for i in range(int(n_test/chunc_size)):
        batch_x = []
        batch_y = []
        for j in range(i*chunc_size, (i+1)*chunc_size-1):
            batch_x.append(X_train[j])
            batch_y.append(Y_train[j])
        test_accuracy += sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
    print("\nAccuracy on test set:", test_accuracy/200)

    
    print()

    plots_x = 1
    plots_y = 1
    fig = plt.figure(figsize=(32, 8))
    samples = sample(range(n_remaining), k=plots_x*plots_y)
    samples.sort()
    counter = 0
    '''
    افزودن تصویر دلخواه جهت تشخیص توسط شبکه عصبی
    '''
    # file="/home/virux/Desktop/rsz_2.png"
    # file="/home/virux/Desktop/rsz_8.png"
    file="/home/virux/Desktop/rsz_7.png"
    img = np.invert(Image.open(file).convert('L')).ravel()
    prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
    print ("Prediction for test image:", np.squeeze(prediction))
    ###end new code

    """
     RemainingSamples.cdb انتخاب عکس به صورت رندم از دیتاست 
    """
    # for i in range(plots_x):
    #     for j in range(plots_y):
    #         prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [X_remaining[samples[counter]]]})

    #         fig.add_subplot(plots_x, plots_y, i*plots_y+j+1)
    #         plt.title('Y_remaining[ ' + str(samples[counter]) + ' ], E: ' + str(list(Y_remaining[samples[counter]]).index(1)) + ', P: ' + str(np.squeeze(prediction)))
    #         plt.imshow(X_remaining[samples[counter]].reshape([32, 32]), cmap='gray')

    #         counter+=1
    # plt.show()
