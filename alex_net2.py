import tensorflow as tf
import preprocess
import numpy as np
import matplotlib.pyplot as plt
from random import sample
import sys

def split_by_percent(data_set):
    num_val = int(len(data_set)*0.2)
    num_test = int(len(data_set)*0.2)
    num_train = len(data_set) - num_val - num_test
    # extract test data
    test_indices = sample(range(len(data_set)), num_test)
    test_data = data_set[test_indices]
    data_set = np.delete(data_set, test_indices, 0)
    # extract validation data
    val_indices = sample(range(len(data_set)), num_val)
    val_data = data_set[val_indices]
    # rest is training data
    train_data = np.delete(data_set, val_indices, 0)
    return train_data, val_data, test_data

def split_train_val_test(split_t): 
    #training
    if split_t == 1 :
        training_data = [i for i in data_set if i.signer_num <=4 ]
        test_data = [i for i in data_set if i.signer_num == 5 ]
        return training_data , test_data
    elif split_t == 2: 
        training_data = [ i for i in data_set if i.signer_num <=3]
        validation_data = [i for i in data_set if i.signer_num == 4 ]
        testing_data = [i  for i in data_set if i.signer_num == 5 ]
        return training_data , validation_data , testing_data
    else: 
        print('Error Split Type : ' , split_t , ' is not supported')

def get_next_batch(batch_index, batch_size, training_images) :
    # create imageset of matrix numInputs x numTotalPixels
    images = training_images[batch_index: batch_index+batch_size]
    labels = training_labels[batch_index: batch_index+batch_size]
    return images, labels

usePercentage = True
usePersonSplit1 = False
usePersonSplit2 = False
have_validation = False

img_height = 28
img_width = 28
batch_size = 20

# Parameters
learning_rate = 0.001

n_epochs = 90
display_step = 10

# Store layers weight & bias
nnInputHeight = 4
nnInputWidth = 4
numKernels1 = 64
numKernels2 = 128
numKernels3 = 256
numNeurons1 = 1024
numNeurons2 = 1024
numImageChannels = 3

# Network Parameters
n_classes = 32 # (0-9, a-z) excluding j,z and o,v
dropout = 0.8 # Dropout, probability to keep units

# read in images
data_set = np.array(preprocess.create_imageset())
print("Created data set.")

image_set = np.array([np.array(i.matrix).reshape(28,28,3) for i in data_set])

if usePercentage:
    training, validation, test = split_by_percent(data_set)
    training_iters = len(training)
    have_validation = True
elif usePersonSplit1:
    training, test = split_train_val_test(1)
    training_iters = len(training)
    print("Split 1")
    print("Training Set  : " , len(training)) 
    print("Testing  Set  : " , len(test))
    have_validation = False
elif usePersonSplit2:
    training, validation, test = split_train_val_test(2)
    training_iters = len(training)
    print("Split 1")
    print("Training    Set  : " , len(training))
    print("Validation  Set  : " , len(validation)) 
    print("Testing     Set  : " , len(test))
    have_validation = True

training_images= np.array([np.array(i.matrix).reshape(28,28,3) for i in training])
training_labels = np.array([np.array(i.label_vec) for i in training])

if have_validation:
    validation_images= np.array([np.array(i.matrix).reshape(28,28,3) for i in validation])
    validation_labels = np.array([np.array(i.label_vec) for i in validation])

test_images= np.array([np.array(i.matrix).reshape(28,28,3) for i in test])
test_labels = np.array([np.array(i.label_vec) for i in test])

print("\n*********INFO**********")
print("test images : ",  test_images.shape)
print("training images : ",  training_images.shape)
print("test labels : ",  test_labels.shape)
print("train labels : ",  training_labels.shape)
print('*********************\n\n\n')

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
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=2)
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv3, k=2)
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    # Apply Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

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
is_in_top5 = tf.cast(tf.nn.in_top_k(predictions=pred, targets=tf.argmax(classes,1), k=5), tf.float32)
top5 = tf.reduce_mean(tf.cast(is_in_top5, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
training_acc = []
training_top5 = []

if have_validation:
    old_val_acc = -1 * sys.maxsize

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1, n_epochs+1):
        print("\n ===== Epoch {} ====\n".format(epoch))
        step = 1
        batch_index = 0

        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            print("Step: " + str(step))
            batch_images, batch_labels  = get_next_batch(batch_index, batch_size, training_images)
            batch_index += batch_size

            # Fit training using batch data
            sess.run(optimizer, feed_dict={inputs: batch_images, classes: batch_labels, keep_prob: dropout})

            # if step % display_step == 0:
                # Calculate batch accuracy
                # acc = sess.run(accuracy, feed_dict={inputs: batch_images, classes: batch_labels, keep_prob: 1.})
                # Calculate batch loss
                # loss = sess.run(cost, feed_dict={inputs: batch_images, classes: batch_labels, keep_prob: 1.})
                # print("Iter {}, Minibatch Loss= {:.6f}, Batch Accuracy= {:.5f}".format(step*batch_size, loss, acc))
            step += 1
        # end of epoch
        # calculate training acc
        acc,epoch_top5 = sess.run([accuracy, top5], 
            feed_dict={inputs: training_images, classes: training_labels, keep_prob: 1.})
        print("Top 1 Accuracy on Training Set = {}".format(acc))
        print("Top 5 Accuracy on Training Set = {}".format(epoch_top5))
        training_acc.append(acc)
        training_top5.append(epoch_top5)

        # calculate validation accuracy every five epochs
        if have_validation and epoch % 5 == 0: 
            curr_val_acc = sess.run(accuracy, feed_dict={inputs: validation_images, classes: validation_labels, keep_prob: 1.})
            print("Accuracy on Validation Set = {}".format(curr_val_acc))
            if(curr_val_acc < old_val_acc):
                # validation accuracy is getting worse
                # so end training to prevent overfitting
                print("Validation accuracy has decreased ({} -> {})".format(old_val_acc, curr_val_acc))
                print("Stopping training after Epoch {} to prevent overfitting.".format(epoch))
                break
            old_val_acc = curr_val_acc
    test_acc, test_top5 = sess.run([accuracy, top5], feed_dict={inputs: test_images, classes: test_labels, keep_prob: 1.})
    print("Top 1 Accuracy on Test Data: ", test_acc)
    print("Top 5 Accuracy on Test Data: ", test_top5)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Epoch')
ax.set_ylabel('Top 1 Accuracy')
plt.title("Top 1 Accuracy on Training Data")
plt.plot(training_acc)
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Epoch')
ax.set_ylabel('Top 5 Accuracy')
plt.title("Top 5 Accuracy on Training Data")
plt.plot(training_top5)
plt.legend()
plt.show()
