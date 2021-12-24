import numpy as np

EPSILON = 1e-10

def _error(actual: np.ndarray, predicted: np.ndarray, mask:np.ndarray):
    """ Simple error """
    return (1-mask) * actual - (1-mask) * predicted

def mse(actual: np.ndarray, predicted: np.ndarray, mask:np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted, mask)))

def rmse(actual: np.ndarray, predicted: np.ndarray, mask:np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted, mask))

def nrmse(actual: np.ndarray, predicted: np.ndarray, mask:np.ndarray):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted, mask) / (actual.max() - actual.min())


METRICS = {
    'mse': mse,
    'rmse': rmse,
    'nrmse': nrmse,
}


def evaluate(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray, metrics=('rmse','mape')):
    
    # reshape the data
    samples = predicted.shape[0]
    actual = actual.reshape(-1, 40, 40)
    predicted = predicted.reshape(-1, 40, 40)
    mask = mask.reshape(-1, 40, 40)
    # Store the results
    results = {}
    # take only imputed values inot consideration
    actual = (1-mask) * actual 
    predicted = (1-mask) * predicted
    
    for i in range(samples):
        # Ignore real nans
        act_sample = actual[i][(~np.isnan(actual[i]))]
        pred_sample = predicted[i][(~np.isnan(actual[i]))]
        mask_sample = mask[i][(~np.isnan(actual[i]))]
        
        for name in metrics:
            try:
                results[name] =+ nrmse(act_sample, pred_sample, mask_sample)/len(predicted)
            except Exception as err:
                results[name] = np.nan
    
                
    return results