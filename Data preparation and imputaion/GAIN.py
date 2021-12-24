# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm

from gain_utils import *
from evaluations import *

def GAIN(miss_data_x, gain_parameters):

    tf.reset_default_graph()

    # Define mask matrix
    data_m = 1-np.isnan(miss_data_x)

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']
    learning_rate = gain_parameters['learning_rate']

    # Other parameters
    no, dim = miss_data_x.shape

    # Hidden state dimensions
    h_dim = int(dim)

    # Normalization
    norm_data, norm_parameters = normalization(miss_data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    ## GAIN architecture   
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape = [None, dim])
    # Mask vector 
    M = tf.placeholder(tf.float32, shape = [None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape = [None, dim])

    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs

    D_W4 = tf.Variable(xavier_init([h_dim, dim]))
    D_b4 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs

    theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4]

    #Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
    G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    G_W3 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b3 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    G_W4 = tf.Variable(xavier_init([h_dim, dim]))
    G_b4 = tf.Variable(tf.zeros(shape = [dim]))

    theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]

    ## GAIN functions
    # Generator
    def generator(x,m):
        # Concatenate Mask and Data
        inputs = tf.concat(values = [x, m], axis = 1) 
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values = [x, h], axis = 1) 
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    ## GAIN structure
    # Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1-M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 

    G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))

    MSE_loss = \
    tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss 

    ## GAIN solver
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=theta_G)

    ## Iterations
    sess = tf.Session()
#     run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    sess.run(tf.global_variables_initializer())
#              , options=run_opts)

    # Start Iterations
    for epoch in range(iterations):
        prog_bar = tqdm(range(int(len(miss_data_x)/batch_size)))
        for step in prog_bar:    
            # Sample batch
            batch_idx = sample_batch_index(no, batch_size)
            X_mb = norm_data_x[batch_idx, :]  
            M_mb = data_m[batch_idx, :]  
            # Sample random vectors  
            Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
            # Sample hint vectors
            H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
            H_mb = M_mb * H_mb_temp

            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
            
            _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                                      feed_dict = {M: M_mb, X: X_mb, H: H_mb})
#                                       , options=run_opts)

            _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss], 
                                                     feed_dict = {X: X_mb, M: M_mb, H: H_mb})
#                                                      , options=run_opts)

            prog_bar.set_description("Epoch({}): D_loss, G_loss ===> {:.3f}, {:.3f}".format(epoch+1, D_loss_curr, G_loss_curr))

    ## Return imputed data      
    Z_mb = uniform_sampler(0, 0.01, no, dim) 
    M_mb = data_m
    X_mb = norm_data_x          
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 

    imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
#                             , options=run_opts)[0]

    imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)  

    # Rounding
    imputed_data = rounding(imputed_data, miss_data_x)  

    return imputed_data