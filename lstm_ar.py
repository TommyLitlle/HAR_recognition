import tensorflow as tf
import numpy as np
from torch import tensor

def batch_norm(input_tensor, config, i):
    
    ## Implementing batch normalisation: this is used out of the residual layers
    # to normalise those output neurons by mean and standard deviation.
    if config.n_layer_in_highway == 0:
        # There is no residual layers, no need for batch_nrm
        return input_tensor
    
    with tf.variable_scope("batch_norm") as scope:
        if i != 0:
            #do not create extra variables for each time step 
            scope.reuse_variables()
            
        # Mean and variance normalisation simply crunched over all axes
        axes =  list(range(len(input.tensor.get_shape())))
        
        mean, variance = tf.nn.moments(input_tensor, axes = axes, shift = None)
        
        stdev = tf.sqrt(variance +0.01)
        
        #Rescaling 
        bn = input_tensor - mean
        bn  /= stdev
        
        #Learnable extra rescaling
        
        # tf.tf.get_variable("relu_fc_weights", initializer=tf.random_normal(mean=0.0, stddev=0.0)
        bn *= tf.get_variable("a_noreg", initializer=tf.random_normal([1], mean=0.5, stddev=0.0))
        bn += tf.get_variable("b_noreg", initializer=tf.random_normal([1], mean=0.0, stddev=0.0))
        # bn *= tf.Variable(0.5, name=(scope.name + "/a_noreg"))
        # bn += tf.Variable(0.0, name=(scope.name + "/b_noreg"))

    return bn

def relu_fc(input_2D_tensor_list, features_len, new_features_len, config):
    
    """
    make a relu fully-connected layer, mainly change the shape of tensor 
    both input and output is a list of tensor
    argument:
      input_2D_tensor_list: list shape is [batch_size, feature_num]
      feature_len: int the initial features length of output_2D_tensor
      config: Config used for weights initializers
    return :
    output_2D_tensor_list lit shape is [batch_size, new_feature_len]
    """
    W = tf.get_variable(
        "relu_fc_weights",
        initializer = tf.random_normal(
            [features_len,new_features_len],
            mean =0.0,
            stddev = float(config.weights_stddev)
        )
    )
    
    b = tf.get_variable(
        'relu_fc_biases_noreg',
        initializer = tf.random_normal(
            [new_features_len],
            mean = float(config.bias_mean),
            stddev = float(config.weights_stddev)
        )
        
    )
    
    
    # intra_timestep multiplication:
    output_2D_tensor_list = [
        tf.nn.relu(tf.matmul(input_2D_tensor,W) +b) \
            for input_2D_tensor in input_2D_tensor_list]
    
    return output_2D_tensor_list