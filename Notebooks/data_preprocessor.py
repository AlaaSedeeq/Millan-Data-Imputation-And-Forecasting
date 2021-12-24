import numpy as np 

def data_preprocessor(data, miss_rate, x_dim, y_dim):
    
    samples, rows, cols, _ = data.shape
    
    startx, starty = cols//2, rows//2
    
    temp_data = np.zeros(shape=(samples, x_dim, y_dim, 1), dtype=np.float32)
    
    for s in range(samples):
        temp_data[s] = data[s][startx-x_dim//2:starty+x_dim//2, startx-x_dim//2:starty+x_dim//2, :]
        
    data_x = np.reshape(temp_data, [samples, x_dim*y_dim]).astype(np.float32)
    
    # Parameters
    no, dim = data_x.shape
    
    # Introduce missing data
    data_m = binary_sampler(1-miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m

def binary_sampler(p, rows, cols):
    '''
    Sample binary random variables.

    Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns

    Returns:
    - binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
    binary_random_matrix = 1*(unif_random_matrix < p)
    return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
    '''
    Sample uniform random variables.

    Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns

    Returns:
    - uniform_random_matrix: generated uniform random matrix.
    '''
    return np.random.uniform(low, high, size = [rows, cols])       