# Necessary packages
import numpy as np
from keras.datasets import mnist
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as  plt
from evaluations import nrmse

def data_preprocessor(data, miss_rate, x_dim, y_dim):

    samples, rows, cols, _ = data.shape
    
    startx, starty = cols//2, rows//2
    
    temp_data = np.zeros(shape=(samples, x_dim, y_dim, 1), dtype=np.float32)
    
    for s in range(samples):
        temp_data[s] = data[s][startx-x_dim//2:starty+x_dim//2, startx-x_dim//2:starty+x_dim//2, :]
    
    # Introduce missing data
    data_m = binary_sampler(1-miss_rate, temp_data.shape)
    miss_data_x = temp_data.copy()
    miss_data_x[data_m == 0] = np.nan
    miss_data_x = miss_data_x.reshape(temp_data.shape)
    
    return temp_data, miss_data_x, data_m

def normalization (data, parameters=None):
  
    samples, rows, cols, _ = data.shape
    
    # Parameters
    norm_data = data.reshape(samples, rows*cols).copy()
    _, dim = norm_data.shape
    

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   

        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  

        norm_parameters = parameters    
      
    return norm_data.reshape(-1, rows, cols, 1), norm_parameters

def rmse_loss (ori_data, imputed_data, data_m):
    
    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)

    # Only for missing values
    nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
    denominator = np.sum(1-data_m)

    rmse = np.sqrt(nominator/float(denominator))

    return rmse

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)

def binary_sampler(p, size):
    unif_random_matrix = np.random.uniform(0., 1., size = size)
    binary_random_matrix = 1*(unif_random_matrix < p)
    return binary_random_matrix

def uniform_sampler(low, high, size):
    return np.random.uniform(low, high, size = size)       

def sample_batch_index(total, batch_size):
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)

# 2D Convolutional Function
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def renormalization (norm_data, norm_parameters):
    
    samples, rows, cols, _ = norm_data.shape
    
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    renorm_data = norm_data.reshape(-1, rows*cols).copy()
    _, dim = renorm_data.shape

    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]

    return renorm_data.reshape(-1, rows, cols, 1)

def rounding (imputed_data, data_x):
    
    samples, rows, cols, _ = imputed_data.shape
    rounded_data = imputed_data.reshape(-1, rows*cols).copy()
    _, dim = rounded_data.shape

    for i in range(dim):
        temp = data_x.reshape(-1, rows*cols)[~np.isnan(data_x.reshape(-1, rows*cols)[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
            
    return rounded_data.reshape(-1, rows, cols, 1)

def show_results(data_x, miss_data_x, data_m, imputed_data, num_examples=1, cmap=None):
    for i in range(num_examples):
        i = np.random.randint(0, len(imputed_data))        
        print('Image : %d'%i)
        # drop original NaN
        actual = data_x[i][np.logical_not(np.isnan(data_x[i]))]
        mask = data_m[i][np.logical_not(np.isnan(data_x[i]))]
        predicted = imputed_data[i][np.logical_not(np.isnan(data_x[i]))]
        miss_data = miss_data_x[i][np.logical_not(np.isnan(data_x[i]))]
        # Only for missing values
        eva = nrmse(actual, predicted, mask)
        print('nrmse : ', eva)
        fig = plt.figure(figsize=(20,8))
        plt.subplot(1,4,1)
        plt.title('Original')
        plt.imshow(data_x[i].reshape(data_x[i].shape), cmap=cmap)
        plt.subplot(1,4,2)
        plt.title('Original With NaN')
        plt.imshow(miss_data_x[i].reshape(data_x[i].shape), cmap=cmap)
        plt.subplot(1,4,3)
        plt.title('NaN Mask')
        plt.imshow((1-data_m[i]).reshape(data_x[i].shape), cmap=cmap)
        plt.subplot(1,4,4)
        plt.title('Imputed')
        plt.imshow(imputed_data[i].reshape(data_x[i].shape), cmap=cmap);
        plt.show()