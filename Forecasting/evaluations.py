import numpy as np 

def nrmse(df_p):
    """
    function to calculate the normalized root mea sqaure error
    """
    return np.sqrt(((df_p['Target']-df_p['Predictions'])**2).sum() / df_p.shape[0])/df_p["Target"].mean()


def mape(df_p):
    """
    function to provide the mean absolute percentage error
    """
    return np.absolute((df_p['Target']-df_p['Predictions']) / df_p["Target"]).sum()/ df_p.shape[0]
