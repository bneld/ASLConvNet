import tensorflow as tf
import preprocess 
import numpy as np
import matplotlib.pyplot as plt

data_set = preprocess.create_imageset() 
print("Created data set.");

training = data_set[:400]
test = data_set[400:501]

# training_images = np.array(np.array(i.matrix).reshape(28,28,3) for i in training)
test_images= np.array([np.array(i.matrix).reshape(28,28,3) for i in test])
training_images= np.array([np.array(i.matrix).reshape(28,28,3) for i in training])

training_labels = np.array([np.array(i.label_vec) for i in training])
test_labels = np.array([np.array(i.label_vec) for i in test])

print("\n*********INFO**********")
print("test images : ",  test_images.shape)
print("training images : ",  training_images.shape)
print("test labels : ",  test_labels.shape)
print("train labels : ",  training_labels.shape)
print('*********************\n\n\n')

img_height = 28
img_width = 28
batch_size = 20
# n_input = 784 * 3 
def get_next_batch(batch_index, batch_size) :
    # create imageset of matrix numInputs x numTotalPixels
    images = training_images[batch_index: batch_index+batch_size]
    labels = training_labels[batch_index: batch_index+batch_size]
    return images, labels

# Parameters
learning_rate = 0.001
training_iters = len(training)
n_epochs = 30
display_step = 10

# Store layers weight & bias
nnInputHeight = 4
nnInputWidth = 4
# nnInputHeight = 85
# nnInputWidth = 77
numKernels1 = 64
numKernels2 = 128
numKernels3 = 256
numNeurons1 = 1024
numNeurons2 = 1024
numImageChannels = 3

# Network Parameters
# n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 36 #  total classes (0-9 digits)
dropout = 0.8 # Dropout, probability to keep units

# tf Graph input
inputs = tf.placeholder(tf.float32, [None, img_height, img_width, numImageChannels]) 
classes = tf.placeholder(tf.float32, [None, n_classes])
predicted_classes = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    # height = 28
    # width = 28
    height = 28
    width = 28
    _X = tf.reshape(_X, shape=[-1, height, width, 3]) #REVISIT

    # image is 680 x 610 x 3
    # Convolution Layer
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # image is 680 x 610 x 64
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=2)
    # image is 340 x 305 x 64
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # image is 340 x 305 x 64
    # Apply Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)
    # image is 340 x 305 x 64

    # Convolution Layer
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # image is 340 x 305 x 128
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=2)
    # image is 170 x 153 x 128
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    # image is 170 x 153 x 128
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)
    # image is 170 x 153 x 128

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # image is 170 x 153 x 256
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv3, k=2)
    # image is 85 x 77 x 256
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    # image is 85 x 77 x 256
    # Apply Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)
    # image is 85 x 77 x 256

    # Fully connected layer
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

#                                       |----- kernel --------| |num layers|
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, numImageChannels, numKernels1])),
    'wc2': tf.Variable(tf.random_normal([3, 3, numKernels1, numKernels2])),
    'wc3': tf.Variable(tf.random_normal([3, 3, numKernels2, numKernels3])),
    'wd1': tf.Variable(tf.random_normal([nnInputHeight*nnInputWidth*numKernels3, numNeurons1])),
    'wd2': tf.Variable(tf.random_normal([numNeurons1, numNeurons2])),
    'out': tf.Variable(tf.random_normal([numNeurons2, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([numKernels1])),
    'bc2': tf.Variable(tf.random_normal([numKernels2])),
    'bc3': tf.Variable(tf.random_normal([numKernels3])),
    'bd1': tf.Variable(tf.random_normal([numNeurons1])),
    'bd2': tf.Variable(tf.random_normal([numNeurons2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = alex_net(inputs, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=classes))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(classes,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
training_acc = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1, n_epochs+1):
        print("\n ===== Epoch {} ====\n".format(epoch))
        step = 1
        batch_index = 0 

        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            print("Step: " + str(step))
            batch_images, batch_labels  = get_next_batch(batch_index, batch_size)
            batch_index += batch_size

            # batch_images = np.reshape(batch_images, (-1, batch_size))
            # batch_labels =  np.reshape(batch_labels, (-1, batch_size))
            print(batch_images.shape)

            # Fit training using batch data
            sess.run(optimizer, feed_dict={inputs: batch_images, classes: batch_labels, keep_prob: dropout})

            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={inputs: batch_images, classes: batch_labels, keep_prob: 1.})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={inputs: batch_images, classes: batch_labels, keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Batch Accuracy= " + "{:.5f}".format(acc))
            step += 1
        # end of epoch
        acc = sess.run(accuracy, feed_dict={inputs: training_images, classes: training_labels, keep_prob: 1.})
        print("Training Accuracy on Full Set = {}".format(acc))
        training_acc.append(acc)

    print("Optimization Finished!")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={inputs: test_images, classes: test_labels, keep_prob: 1.}))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
plt.title("Training Accuracy")
plt.plot(training_acc)
plt.legend()
plt.show()





