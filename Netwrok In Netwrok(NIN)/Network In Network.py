import tensorflow as tf
import numpy as np
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def CNN_Layers(x):

    x_image = x

    # MLP Layer 1
    W1 = tf.Variable(tf.random_normal([5, 5, 3, 192], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
    L1 = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
    L1 = tf.nn.relu(L1)

    W2 = tf.Variable(tf.random_normal([1, 1, 192, 160], stddev=0.05, dtype=tf.float32))
    b2 = tf.Variable(tf.constant(0, shape=[160], dtype=tf.float32))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2
    L2 = tf.nn.relu(L2)

    W3 = tf.Variable(tf.random_normal([1, 1, 160, 96], stddev=0.05, dtype=tf.float32))
    b3 = tf.Variable(tf.constant(0, shape=[96], dtype=tf.float32))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3
    L3 = tf.nn.relu(L3)

    L3 = tf.nn.max_pool(L3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    L3 = tf.nn.dropout(L3, 0.5)

    # MLP Layer 2
    W4 = tf.Variable(tf.random_normal([5, 5, 96, 192], stddev=0.05, dtype=tf.float32))
    b4 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME') + b4
    L4 = tf.nn.relu(L4)

    W5 = tf.Variable(tf.random_normal([1, 1, 192, 192], stddev=0.05, dtype=tf.float32))
    b5 = tf.Variable(tf.constant(0, shape=[192], dtype=tf.float32))
    L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME') + b5
    L5 = tf.nn.relu(L5)

    W6 = tf.Variable(tf.random_normal([1, 1, 192, 192], stddev=0.05, dtype=tf.float32))
    b6 = tf.Variable(tf.constant(0, shape=[192], dtype=tf.float32))
    L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME') + b6
    L6 = tf.nn.relu(L6)

    L6 = tf.nn.max_pool(L6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    L6 = tf.nn.dropout(L6, 0.5)

    # MLP Layer 3
    W7 = tf.Variable(tf.random_normal([3, 3, 192, 192], stddev=0.05, dtype=tf.float32))
    b7 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
    L7 = tf.nn.conv2d(L6, W7, strides=[1, 1, 1, 1], padding='SAME') + b7
    L7 = tf.nn.relu(L7)

    W8 = tf.Variable(tf.random_normal([1, 1, 192, 192], stddev=0.05, dtype=tf.float32))
    b8 = tf.Variable(tf.constant(0, shape=[192], dtype=tf.float32))
    L8 = tf.nn.conv2d(L7, W8, strides=[1, 1, 1, 1], padding='SAME') + b8
    L8 = tf.nn.relu(L8)

    W9 = tf.Variable(tf.random_normal([1, 1, 192, 10], stddev=0.05, dtype=tf.float32))
    b9 = tf.Variable(tf.constant(0, shape=[10], dtype=tf.float32))
    L9 = tf.nn.conv2d(L8, W9, strides=[1, 1, 1, 1], padding='SAME') + b9
    L9 = tf.nn.relu(L9)
    output = tf.nn.avg_pool(L9, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')

    output = tf.reshape(output, [-1, 1 * 1 * 10])
    logits = output
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

(x_train, y_train), (x_test, y_test) = load_data()
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

y_pred, logits = CNN_Layers(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch = next_batch(128, x_train, y_train_one_hot.eval())

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

            print("Epoch: %d, Accuracy: %f, loss: %f" % (i, train_accuracy, loss_print))
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    test_batch = next_batch(10000, x_test, y_test_one_hot.eval())
    print("Accuracy: %f" % accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0}))

