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
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 100
        self.grad_clip =15
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

def LSTM_Network(_X,config):
    """
    Function returns a Tensorflow with two staceked LSTM cells
    Two LSTM cells are stacked which adds deepness to the neural network.
    Note, some code of this notebook is inspired from an slightly differnt 
    RNN architecture used on another dataset, some of the credits goes to
    "aymericdamien".
    Args:
        _X:     ndarray feature matrix, shape: [batch_size, time_steps, n_inputs]
        config: Config for the neural network.
    Returns:
        This is a description of what is returned.
    Raises:
        KeyError: Raises an exception.
      Args:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    # (((Note: this step could be greatly optimized by shaping the dataset once
    #input shape: (batch_size, n_steps, n_input)
    
    _X = tf.nn.dropout(_X, 0.8)
    _X = tf.transpose(_X, [1,0,2]) #permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, config.n_inputs])
    
    #new shape: (n_steps*batch_size,n_input)
    
    #Linear activation 
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)
    
    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.999, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.999, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]
    
    
    #Linear activation 
    return tf.matmul(lstm_last_output,config.W['output']) + config.biases['output']






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

   

    pred_Y = LSTM_Network(X, config)


    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    
   
  
    cost = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) 

    
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.grad_clip)
    train_op = tf.train.AdamOptimizer(config.learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    # optimizer = tf.train.AdamOptimizer(
        #learning_rate=config.learning_rate).minimize(cost)
 
    
  
   
    
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
    for i in range(config.training_epochs+900):
         
        
        #train by batch
        batch_features, batch_labels = train_data.next_batch(config.batch_size)
                
        #input dictionary with dropout of 50%

        
        sess.run(optimizer, feed_dict={X: batch_features,
                                           Y: batch_labels})            
           
        
        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run(
            [pred_Y, accuracy, cost],
            feed_dict={
                X: X_test,
                Y: y_test
            }
        )
      
      
        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out) +
              " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)
     
   
        

    print("") 
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")
   

