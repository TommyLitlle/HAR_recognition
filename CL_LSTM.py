import tensorflow as tf

import numpy as np
from sklearn.metrics import classification_report

# Load "X" (the neural network's training and testing inputs)

from Load_data import read_data_sets

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.002
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 500
        self.layers =2
        self.grad_clip =5

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
        self.n_hidden = 32  # nb of neurons inside the neural network
        self.n_classes = 18  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)



def LSTM_Network(X,config):
   
    def get_a_cell(n_hidden):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            return lstm
   
    cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(config.n_hidden) for _ in range(config.layers)]
            )
   
    init_state = cell.zero_state(config.batch_size, dtype=tf.float32)
   
    output, states = tf.nn.dynamic_rnn(cell,X, dtype=tf.float32)
    
    output_flattened = tf.reshape(output,[-1,config.n_hidden])
    
    output_logits = tf.add(tf.matmul(output_flattened, config.W['output']), config.biases['output'])
    
    output_all = tf.nn.softmax(output_logits)
    
    
    output_reshaped = tf.reshape(output_all,[-1,config.n_steps,config.n_classes])
    
    output_second = tf.gather(tf.transpose(output_reshaped,[1,0,2]), config.n_steps - 2) 
    
    output_last = tf.gather(tf.transpose(output_reshaped,[1,0,2]), config.n_steps - 1)  
    
   
    
    #Linear activation 
    return output_last,output_second, output_reshaped






if __name__ == "__main__":

 
    # data set information 
    
    data=read_data_sets()
    train_data=data.train
    test_data = data.test
    X_train = train_data.data
    X_test = test_data.data
    y_test = test_data.labels
    
    # -----------------------------------
    # Step 2: define parameters for model
    # -----------------------------------

    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    # ------------------------------------------------------
    # Step 3: Let's get serious and build the neural network
    # ------------------------------------------------------

    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    y_steps = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_Y, pred_second, y_all = LSTM_Network(X, config)



    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    
    cost = 0
    
    for i in range(24):
        y=tf.gather(tf.transpose(y_all,[1,0,2]), i)  
        cost +=tf.reduce_mean(-Y*tf.log(y+ 1e-8))
 
    #cost =tf.reduce_mean(-Y*tf.log(pred_Y+ 1e-7)*(1 - pred_Y))
        
    
    #cost = tf.reduce_mean(
    #  tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) + l2

    
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.grad_clip)
    train_op = tf.train.AdamOptimizer(config.learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    # optimizer = tf.train.AdamOptimizer(
        #learning_rate=config.learning_rate).minimize(cost)
 
    
    prediction = tf.add(pred_Y,pred_second)
    output = tf.argmax(pred_Y, 1)
    label =  tf.argmax(Y, 1)
    
    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))



    # --------------------------------------------
    # Step 4: Hooray, now train the neural network
    # --------------------------------------------

    # Note that log_device_placement can be turned ON but will cause console spam with RNNs.
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs+700):
         
        
        #train by batch
        batch_features, batch_labels = train_data.next_batch(config.batch_size)
                
        #input dictionary with dropout of 50%
       
        
        sess.run(optimizer, feed_dict={X: batch_features,
                                           Y: batch_labels
                                           })            
           
       
        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out, label_, y_put = sess.run(
            [pred_Y, accuracy, cost, label, output],
            feed_dict={
                X: X_test,
                Y: y_test
            }
        )
      
        print(classification_report(label_, y_put, digits=4))
       
        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out) +
              " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)
        
    print(classification_report(label_, y_put, digits=4))     
     
    print("") 
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")
   

