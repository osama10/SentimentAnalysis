import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)
from  Create_Sentiment_Featuresets import  create_feature_set_and_labels
train_x, train_y, test_x, test_y = create_feature_set_and_labels('pos.txt','neg.txt')
#3 hidden layers each having 500 different nodes
n_nodes_hl1 =  500
n_nodes_hl2 =  500
n_nodes_hl3 =  500

n_classes = 2
batch_size = 100

#height x width
x= tf.placeholder('float' ,[None , len(train_x[0])])
y= tf.placeholder('float')

def neural_network_model(data):

    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input * weight) + bias

    l1 = tf.add(tf.matmul(data , hidden_layer_1['weights']) , hidden_layer_1['biases'] )
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']) , hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']) , hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_net(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs =  10
    with tf.Session( ) as sess :
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end =  i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer , cost] , feed_dict={x: batch_x , y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch' , epoch , 'completed out of' , hm_epochs , 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:' , accuracy.eval({x:test_x , y:test_y}))

train_neural_net(x)

print('done')