# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm

from conv_gain_utils import *
from evaluations import *

def Conv_GAIN(miss_data_x, conv_gain_parameters):
    '''
    Impute missing values in data

    Args:
    - data: original data with missing values
    - gain_parameters: GAIN network parameters:
    - batch_size: Batch size
    - hint_rate: Hint rate
    - alpha: Hyperparameter
    - iterations: Iterations

    Returns:
    - imputed_data: imputed data 
    '''
    
    # Define mask matrix
    data_m = 1-np.isnan(miss_data_x)

    # System parameters
    batch_size = conv_gain_parameters['batch_size']
    hint_rate = conv_gain_parameters['hint_rate']
    alpha = conv_gain_parameters['alpha']
    iterations = conv_gain_parameters['iterations']
    learning_rate = conv_gain_parameters['learning_rate']

    # Other parameters
    no, rows, cols, _ = miss_data_x.shape
    g_last_w = int(np.ceil(rows/4)*np.ceil(3*cols/4)*64)
    d_last_w = int(np.ceil(rows/4)*np.ceil(2*cols/4)*64)    
    
    # Hidden state dimensions
    h_dim = (rows, cols, 1)

    # Normalization
    norm_data, norm_parameters = normalization(miss_data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    # Reset the default graph
    tf.reset_default_graph()

    ### GAIN architecture   
    ## Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape = [None, rows, cols, 1])
    # Mask vector
    M = tf.placeholder(tf.float32, shape = [None, rows, cols, 1])
    # Hint vector
    H = tf.placeholder(tf.float32, shape = [None, rows, cols, 1])
    # Noise vector
    Z = tf.placeholder(tf.float32, shape = [None, rows, cols, 1])

    ## D Weights and Biases
    # Input ==> (rows, 2*cols)
    D_weights = {
        # Convolution Layers
        'c1': tf.Variable(xavier_init((3,3,1,32)), name='W1'), 
        'c2': tf.Variable(xavier_init((3,3,32,64)), name='W2'),

        # Dense Layers
        'd1': tf.Variable(xavier_init((d_last_w, d_last_w//3)), name='W3'),
        'out': tf.Variable(xavier_init((d_last_w//3, rows*cols)), name='W4'), 
    }
    D_biases = {
        # Convolution Layers
        'c1': tf.Variable(xavier_init((32,)), name='B1'),
        'c2': tf.Variable(xavier_init((64,)), name='B2'),

        # Dense Layers
        'd1': tf.Variable(xavier_init((d_last_w//3,)), name='B3'),
        'out': tf.Variable(xavier_init((rows*cols,)), name='B4'),
    }

    theta_D = np.concatenate(list((list(D_weights.values()), list(D_biases.values()))))

    ## G Weights and Biases
    # Input ==> (rows, 3*cols)
    G_weights = {
        # Convolution Layers
        'c1': tf.Variable(xavier_init((3,3,1,32)), name='W1'), 
        'c2': tf.Variable(xavier_init((3,3,32,64)), name='W2'),

        # Dense Layers
        'd1': tf.Variable(xavier_init((g_last_w, g_last_w//3)), name='W3'),
        'out': tf.Variable(xavier_init((g_last_w//3, rows*cols)), name='W4'),
    }
    G_biases = {
        # Convolution Layers
        'c1': tf.Variable(xavier_init((32,)), name='B1'),
        'c2': tf.Variable(xavier_init((64,)), name='B2'),

        # Dense Layers
        'd1': tf.Variable(xavier_init((g_last_w//3,)), name='B3'),
        'out': tf.Variable(xavier_init((rows*cols,)), name='B4'),
    }

    theta_G = np.concatenate(list((list(G_weights.values()), list(G_biases.values()))))

    # 2D Convolutional Function
    def conv2d(x, W, b, strides=2):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)
        return x

    def generator(x, m, z):
        # Concatenate Mask and Data
        inputs = tf.concat(values = [x, m, z], axis = 1)
        # Convolution layers
        #First Layer
        conv1 = conv2d(inputs, G_weights['c1'], G_biases['c1']) # [rows, 3*cols, 1] ==> [rows/2, 3*cols/2, 32]
#         pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
#         conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
        #Second Layer
        conv2 = conv2d(conv1, G_weights['c2'], G_biases['c2']) # [rows/2, 3*cols/2, 32] ==> [rows/4, 3*cols/4, 64]
#         conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
#         pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 
        # Flatten Layer
        flat = tf.reshape(conv2, [-1, G_weights['d1'].get_shape().as_list()[0]]) # [(rows/4)*(3*cols/4)*64]
        # Fully connected layer
        fc1 = tf.nn.relu(tf.add(tf.matmul(flat, G_weights['d1']), G_biases['d1']))
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.add(tf.matmul(fc1, G_weights['out']) , G_biases['out'])) # [rows, cols]
        G_prob = tf.reshape(G_prob, [-1, rows, cols, 1])
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # Concatenate Mask and Data
        inputs = tf.concat(values = [x, h], axis = 1) 
        # Convolution layers
        #First Layer
        conv1 = conv2d(inputs, D_weights['c1'], D_biases['c1']) # 
#         pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 
#         conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
        #Second Layer
        conv2 = conv2d(conv1, D_weights['c2'], D_biases['c2']) # 
#         conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
        # Flatten Layer
        flat = tf.reshape(conv2, [-1, D_weights['d1'].get_shape().as_list()[0]])
        # Fully connected layer
        fc1 = tf.nn.relu(tf.add(tf.matmul(flat, D_weights['d1']), D_biases['d1']))
        # MinMax normalized output
        D_logit = tf.add(tf.matmul(fc1, D_weights['out']) ,D_biases['out']) # [728]
        D_prob = tf.nn.sigmoid(D_logit)
        D_prob = tf.reshape(D_prob, [-1, rows, cols, 1])
        return D_prob

    ## GAIN structure
    # Generator
    G_sample = generator(X, M, Z)

    # Combine with observed data(real obs + missed after generation)
    Hat_X = X * M + G_sample * (1-M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8)) 
    
    G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))

    MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss 

    ## GAIN solver
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=theta_D.tolist())
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=theta_G.tolist())

    # Start the session
    sess = tf.Session()
    # Initialize the global parameters 
    sess.run(tf.global_variables_initializer())

    # Start Iterations
    for epoch in range(iterations):
        prog_bar = tqdm(range(int(len(miss_data_x)/batch_size)))
        for step in prog_bar:  
            # Sample batch
            batch_idx = sample_batch_index(no, batch_size)
            X_mb = norm_data_x[batch_idx, :] 
            X_mb = np.reshape(X_mb, (batch_size, rows, cols, 1))
            M_mb = data_m[batch_idx, :]  
            M_mb = np.reshape(M_mb, (batch_size, rows, cols, 1))
            # Sample random vectors  
            Z_mb = uniform_sampler(0, 0.01, (batch_size, rows, cols, 1)) 
            Z_mb = np.reshape(Z_mb, (batch_size, rows, cols, 1))
            # Sample hint vectors
            H_mb_temp = binary_sampler(hint_rate, (1, rows, cols, 1))
            H_mb = M_mb * H_mb_temp
            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
            Z_mb = np.reshape(Z_mb, X_mb.shape)
            
            _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                                      feed_dict = {M: M_mb, X: X_mb, H: H_mb, Z: Z_mb})

            _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss],
                                                     feed_dict = {X: X_mb, M: M_mb, H: H_mb, Z: Z_mb})

            prog_bar.set_description("Epoch({}): D_loss, G_loss ===> {:.3f}, {:.3f}".format(epoch+1, D_loss_curr, G_loss_curr))

    # Return imputed data
    # Z vector
    Z_mb = uniform_sampler(0, 0.01, (no, rows, cols, 1)) 
    Z_mb = np.reshape(Z_mb, (no, rows, cols, 1))
    # M vector
    M_mb = data_m
    M_mb = np.reshape(M_mb, (-1, rows, cols, 1))
    # X vector
    X_mb = norm_data_x     
    X_mb = np.reshape(X_mb, (-1, rows, cols, 1))
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

    imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb, Z:Z_mb})[0]
    imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data

    imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)  

    # Rounding
    imputed_data = rounding(imputed_data, miss_data_x) 
    
    return imputed_data