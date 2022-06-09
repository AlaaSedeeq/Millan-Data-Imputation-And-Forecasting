import numpy as np
import pandas as pd
from scipy import stats
import datetime as dt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
# from sklearn.impute import KNNImputer


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
        # Ignore original nans
        act_sample = actual[i][(~np.isnan(actual[i]))]
        pred_sample = predicted[i][(~np.isnan(actual[i]))]
        mask_sample = mask[i][(~np.isnan(actual[i]))]
        
        for name in metrics:
            try:
                results[name] =+ METRICS[name](act_sample, pred_sample, mask_sample)/len(predicted)
            except Exception as err:
                results[name] = np.nan
    return results


def imputation(data_x:np.ndarray, miss_data_x:np.ndarray, data_m:np.ndarray, method:str, n_neighbors=2, max_iter=1):
    
    def fillnan_eval(real:np.ndarray, with_nan:np.ndarray, mask:np.ndarray, method:str, metrics=('rmse','nrmse')):
        """ 
        Filling NaN values
        methods: Mean, Mode, Median, Mice, LVO, NVO, or Interpolation 
        """
        
        # ignore the original NaNs
        with_nan = with_nan[np.argwhere(~np.isnan(real))].flatten()
        mask = mask[np.argwhere(~np.isnan(real))].flatten()
        real = real[np.argwhere(~np.isnan(real))].flatten()
        
        # calculate imputed values
        temp = with_nan[np.argwhere(~np.isnan(with_nan))]
        
        # choose the imputaion method
        if method.lower()=='mean':
            value = np.mean(temp)
            imputed = pd.Series(with_nan).fillna(value)

        elif method.lower()=='mode':
            value = np.array(stats.mode(temp)[0])[0][0]
            imputed = pd.Series(with_nan).fillna(value)
        
        elif method.lower()=='median':
            value = np.nanmedian(temp)
            imputed = pd.Series(with_nan).fillna(value)
        
        elif method.lower()=='lvo':
            imputed = pd.Series(with_nan).fillna(method='ffill')

        elif method.lower()=='nvo':
            imputed = pd.Series(with_nan).fillna(method='bfill')
        
        elif method.lower()=='inter':
            df = pd.DataFrame(with_nan)
            df['time'] = pd.RangeIndex(start=1383260400000, stop=1383260400000+len(df)*10000*60, step=10000*60)
            df['time'] = df['time'].apply(lambda t: dt.datetime.fromtimestamp(t / 1e3)-\
                                                    dt.timedelta(hours = 1))
            df['time'] = pd.to_datetime(df['time'])
            df['minute'] = df['time'].dt.minute
            df= df.set_index('time')
            imputed = df.groupby('minute').apply(lambda x : 
                                                 x.interpolate(method='linear', 
                                                               linit_area='indside'))[0]
        return real, with_nan, mask, imputed
    
    if method.lower() in ['mean','mode','median','lvo','nvo']:
        # store the results
        results = {'nrmse':0}
        # iterate over the data
        for i in range(0, data_x.shape[1]):
            for j in range(0, data_x.shape[2]):
                real, with_nan, mask, imputed = fillnan_eval(data_x[:,i,j], miss_data_x[:,i,j], data_m[:,i,j], method)
                results['nrmse'] += nrmse(real, imputed, mask)/len(imputed*(1-mask))
                
    elif method.lower() in ['inter','interpolation']:
        method = 'inter'
        # store the results
        results = {'nrmse':0}
        # iterate over the data
        for i in range(0, data_x.shape[1]):
            for j in range(0, data_x.shape[2]):
                real, with_nan, mask, imputed = fillnan_eval(data_x[:,i,j], miss_data_x[:,i,j], data_m[:,i,j], method)
                results['nrmse'] += nrmse(real, imputed, mask)/len(imputed*(1-mask))

    elif method.lower()=='mice':
        # store the results
        results = 0
        # Convert the data into multivariate
        miss_x = pd.DataFrame()
        real_x = pd.DataFrame()
        for i in range(data_x.shape[1]):
            for j in range(data_x.shape[2]):
                miss_x[i*j] = pd.Series(miss_data_x[:,i,j].flatten())
                real_x = data_x[:,i,j].flatten()
                mask = data_m[:,i,j].flatten()
                # define imputer
                lr = LinearRegression()
                imp = IterativeImputer(estimator=lr,missing_values=np.nan, 
                                       max_iter=max_iter, verbose=0, 
                                       imputation_order='ascending')
                # fit on the dataset and transform
                imputed = imp.fit_transform(miss_x).flatten()
                results += nrmse(real_x, imputed, mask)/len(imputed*(1-mask))
            return results
        
    elif method.lower()=='knn':
        results = {'nrmse':0}
        # define the data
        for i in range(0, data_x.shape[1]):
            for j in range(0, data_x.shape[2]):
                real, with_nan, mask = data_x[:,i,j], miss_data_x[:,i,j], data_m[:,i,j]
                # define imputer                
                imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform', metric='nan_euclidean')
                # fit on the dataset and transform
                imputed = imputer.fit_transform(with_nan.reshape(-1, 1))
                results['nrmse'] += nrmse(real, imputed, mask)/len(imputed*(1-mask))
                
        return results

    else:
        raise(Exception('Only  mean, mode, median, LVO, NVO, mice and knn'))
        
    return results