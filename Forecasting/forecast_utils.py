#Basic libraries
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt 
import warnings
import datetime
from warnings import filterwarnings, catch_warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)               

#statistics libraries
import scipy
from  scipy.stats import  boxcox
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR, SimpleExpSmoothing, ExponentialSmoothing
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

#DL libraries
import keras
from keras import Model
import tensorflow as tf
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, concatenate

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly 
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
import cufflinks as cf
import plotly.figure_factory as ff 
from plotly.offline import iplot
from plotly import tools
from evaluations import nrmse, mape


# To convert time column into a datetime
def convert(t):
    index = dt.datetime.fromtimestamp(t / 1e3) - dt.timedelta(hours = 1)
    index = pd.to_datetime(index)
    return index

def neighbor_cells(data:np.ndarray, neighbor_radius:int, cell_center:list, num_samples=None, uni_variant=True):
    if not num_samples:
        num_samples = len(data)
    # the cell center coordination.
    cell_center = np.array(cell_center)
    # check if cell center coordination is only one value.
    if len(cell_center) == 1:
            # ensure it is in the range of data's boundaries.
            # if not, reject it.
            if cell_center[0] > data.shape[1]-1:
                raise(Exception('Grid target must be a tuple has the position of the target cell within the grid'))
            # if it in the range of data's boundaries, take it as x and y.
            else:
                row_number, column_number = cell_center[0], cell_center[0]
                return row_number, column_number
    # check if cell center coordination is an array of more than two values.
    elif len(cell_center) != 2:
        raise(Exception('Cell target must be a tuple has the position of the target cell within the grid'))
    # check if cell center coordination is an array of only two values.
    else:
        # ensure it is in the range of data's boundaries.
        # if not, reject it.
        if (cell_center[0]>data.shape[1]-1) or (cell_center[1]>data.shape[2]-1):
            raise(Exception('Cell target must be a tuple has the position of the target cell within the grid'))
        # if each value is in the range of data's boundaries, take them as x and y.
        else:
            row_number, column_number = cell_center[0], cell_center[1]
        
    # the output data frame
    multi_grids_data = pd.DataFrame()
    multi_grids_data['time'] = pd.RangeIndex(start=0, stop=num_samples*10000, step=10000)
    multi_grids_data["time"] = multi_grids_data["time"].apply(lambda t: dt.datetime.fromtimestamp(t / 1e3) - \
                                                                        dt.timedelta(hours = 1))
    multi_grids_data["time"] = pd.to_datetime(multi_grids_data["time"])
    multi_grids_data.set_index('time')
    ## if Uni-Variant (Forcasting the center cell with the neighbors as exogenous variables)
    if uni_variant:
        # get the coordinations of all center cell's neighbors
        values = np.array([[(i+1,j+1) if (i+1==column_number and j+1==row_number) else (np.nan, np.nan)
                                      for j in range(column_number-1-neighbor_radius, column_number+neighbor_radius)]
                                          for i in range(row_number-1-neighbor_radius, row_number+neighbor_radius)])
        # drop nan values (out of original data boundaries)
        if np.isnan(values).max():
            values = np.array(list(filter(lambda v: v==v, values.flatten()))).astype(int)
        # reshape the values into (x, y)
        exo_cells_idx = values.reshape(-1, 2)
        # create column feature for each cell and put them all in one data frame
        for idx in exo_cells_idx:
            if idx[0]!=column_number and idx[1]!=row_number:
                multi_grids_data['Grid({},{})'.format(idx[0],idx[1])] = data[:num_samples, idx[0], idx[1], :].flatten()
        # center cell is the target and the neighbors are exogenous variables
        multi_grids_data['Target'] = data[:num_samples, row_number, column_number, :].flatten()
    ## if Muti-Variant (Forcasting the center cell with the neighbors as exogenous variables)
    else:
        # get the coordinations of all cell's neighbors
        values = np.array([[(i+1,j+1) for j in range(column_number-1-neighbor_radius, column_number+neighbor_radius)]
                                          for i in range(row_number-1-neighbor_radius, row_number+neighbor_radius)])
        # drop nan values (out of original data boundaries)
        if np.isnan(values).max():
            values = np.array(list(filter(lambda v: v==v, values.flatten()))).astype(int)
        # reshape the values into (x, y)
        exo_cells_idx = values.reshape(-1, 2)
        # create column feature for each cell and put them all in one data frame
        for idx in exo_cells_idx:
            multi_grids_data['Grid({},{})'.format(idx[0],idx[1])] = data[:num_samples, idx[0], idx[1], :].flatten()
            
    return multi_grids_data

def divide_data(iter_num, data, t_period , p_period ):
    """
    function to divide the data set into training and prediction sets
    """
    t_periods = {"week": 144*7, "month": 144*30}
    p_periods = {"day": 144, "week": 144*7}
    
    # For training set
    train_period = t_periods[t_period]
    start_t = p_periods[p_period]*iter_num
    end_t = start_t + train_period
#     return start_t, end_t, data
    X_train = data.iloc[start_t:end_t]
    
    # For prediction set
    predict_period = p_periods[p_period]
    start_p = end_t
    end_p = start_p + predict_period
    X_predict = data.iloc[start_p:end_p]
    
    full_predict = data.iloc[start_p:end_p]
    
    return [X_train.set_index('time'), X_predict.set_index('time'), full_predict.set_index('time')]

def EXP_Smoothing(X_train, forecast_data, args):
    
    trend = args['trend']
    seasonal = args['seasonal']
    seasonal_periods = args['seasonal_periods']
    errors = {'nrmse':None, 'mape':None}
    forecasting_steps = len(forecast_data)

    with catch_warnings():
        filterwarnings("ignore")
        EXP_smooth = ExponentialSmoothing(X_train, trend='mul',seasonal='mul',seasonal_periods=144).fit()
        EXP_smooth_pred = EXP_smooth.forecast(len(forecast_data))
    
    forecast_data['Predictions'] = EXP_smooth_pred.values
    
    errors['nrmse'] = nrmse(forecast_data['Target'].values, forecast_data['Predictions'].values)
    errors['mape'] = mape(forecast_data['Target'].values, forecast_data['Predictions'].values)

    return forecast_data, errors


def create_acf_pacf(series, series_name=''):
    
    fig, axes = plt.subplots(1, 2, figsize=(20,5))
        
    sm.graphics.tsa.plot_acf(series, 
                             title='Autocorrelation for %s'%(series_name),
                             lags=40, ax=axes[0])
    
    sm.graphics.tsa.plot_pacf(series,
                              title='Partial Autocorrelation for %s'%(series_name),
                              lags=40, ax=axes[1])
    
    plt.show()
    
class MLP_Forecasting:
    """
    This class creates and train a MLP model for Univariate Multi-Step Forecasting.
    ...
    Attributes
    ----------
    data : Pandas dataframe
    Data to fit the model with.
    """
    
    def __init__(self, training):
        
        self.training_data = training
        self.sequence = training.values
        self.index = training.index
        
    # split a univariate sequence into samples
    @tf.autograph.experimental.do_not_convert
    def split_sequence(self, data, train_test=False, scale=False):

        """
        Docstring:
        This function transforms the raw dataset(TS) into input(Features) and output.
        ...
        Attributes
        ----------
        sequence : Pandas DataFrame
        The data you want to transform.
        
        w_size : integar number
        The size of applied window.
        
        Returns
        ----------
        Two numpy arrays(X, y).
        """
        self.sequence = data
        #create our dataset
        dataset = tf.data.Dataset.from_tensor_slices(self.sequence)
        #apply window function to the dataset to convert the sequence to features and target
        dataset = dataset.window(self.w_size, shift=1, drop_remainder=True)
        #flatten the dataset to numpy array to start dealing with it
        dataset = dataset.flat_map(lambda window: window.batch(self.w_size))
        #shuffling and splitting the dataset into X,y
        dataset = dataset.map(lambda window: (window[:self.n_steps_in], window[-self.n_steps_out:]))
        
        self.X = np.array([x.numpy() for x, _ in dataset])
        self.y = np.array([y.numpy() for _, y in dataset])
        
        #scaling the data 
        if self.is_scale:
            self.X, self.y = self.scale_data(self.X, self.y)
        
        # Split data using train proportion of (70% - 30%)
        if train_test:
            self.split_point = int(0.7 * len(self.X)) 
            self.X_train, self.X_test = self.X[:self.split_point], self.X[self.split_point:]
            self.y_train, self.y_test = self.y[:self.split_point], self.y[self.split_point:]
            
            return self.X_train, self.X_test, self.y_train, self.y_test
        
        return self.X, self.y
    
    def scale_data(self, X, y):

        self.scaler_features = MinMaxScaler().fit(X.reshape(-1, 1))
        self.scaled_features = self.scaler_features.transform(X.reshape(-1, 1)).reshape(np.shape(X))

        self.scaler_label = StandardScaler().fit(np.array(y.reshape(-1, 1)))
        self.scaled_label = self.scaler_label.transform(y.reshape(-1, 1)).reshape(np.shape(y))
        
        return self.scaled_features, self.scaled_label
                          
    def build_dense_layers(self, n_steps_in, n_steps_out, units, scale=False, epochs=100, lr=0.03, batch=256, verbose=1):
        """
        This function creates and train a model for Univarite TS.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
        The data you want to see information about
        
        Returns
        ----------
        A Pandas Series contains the missing values in descending order
        """

        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.is_scale = scale
        self.units = units
        self.n_steps_in =  n_steps_in
        self.n_steps_out = n_steps_out
        self.w_size = self.n_steps_in + self.n_steps_out
        
        # split the sequence into samples
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_sequence(self.sequence, train_test=True, scale=True)
        
        callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0,
                                                 patience=5,
                                                 verbose=0, 
                                                 mode='auto')
        #define the optimizer
        opt = Adam(learning_rate=self.lr)
        
        # Build the model
        self.model = Sequential()
        self.model.add(Dense(units=self.units, input_dim=self.n_steps_in, kernel_initializer='normal', activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=self.units/2, kernel_initializer='normal', activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=self.units/2, kernel_initializer='normal', activation='relu')) 
        self.model.add(Dense(self.n_steps_out))
        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer=opt)

        self.history = self.model.fit(x=self.X_train, y=self.y_train, 
                                      validation_data=(self.X_test, self.y_test),
                                      batch_size=self.batch, 
                                      callbacks=callback,
                                      epochs=self.epochs, 
                                      verbose=verbose)

        return self.model
    
    def show_train_test_results(self):
        """
        This function plots the results of training the Univarite model.
        
        Returns
        ----------
        A Plotly plot.
        """
        temp = pd.DataFrame({'Training':self.history.history['loss'],
                             'Validation':self.history.history['val_loss']})
        
        fig = px.line(temp[['Training','Validation']])
        fig.show()

    def forecast(self, test):
        from evaluations import nrmse, mape
        """
        This function uses the trained model to forecast the test data and plot the results.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
        The data you want to forecast.

        Returns
        ----------
        A Plotly plot.
        """
        #lost data
        self.test_index = test.index
        test = test.append(self.training_data.iloc[-(self.w_size-1):]).sort_index()
        self.test = test
        self.sequence = test.values
        self.forecasting_data = self.test
        errors = {'nrmse':None, 'mape':None}

        self.test_X, self.test_y = self.split_sequence(self.forecasting_data)  
        
        if self.is_scale:
            self.actual_data = self.scaler_label.inverse_transform(self.test_y)[:,:,0]
            self.pred_data = self.model.predict(self.test_X)
            self.pred_data = self.scaler_label.inverse_transform(self.pred_data)
            
        else:
            self.actual_data = self.test_y[:,:,0]
            self.pred_data = self.model.predict(self.test_X)
        
        errors['nrmse'] = nrmse(self.actual_data, self.pred_data)
        errors['mape'] = mape(self.actual_data, self.pred_data)
        
        results = test[(self.w_size-1):]
        
        results['Predictions'] = self.pred_data[-1].flatten()
        
#         results['Predictions'] = Multi_Step_MLP.pred_data[-1]

#         results = pd.DataFrame({'Actuals':self.actual_data[-1], 'Predictions':self.pred_data[-1]}, index=test.index)
#         results = {i[0]:(i[1], i[2]) for i in [v for v in zip(test.index, self.actual_data, self.pred_data)]}
        
        return results, errors
    
def plot_forecasting(real:pd.DataFrame, Results:dict, legend:list, models:list, w, day_week='Day', previous_steps=300):
    
    if len(models)==1:
        data = Results['%s'%models[0]]['%s'%day_week]['%d'%w]['results'].rename(columns={'Predictions':'%s'%models[0]})\
               .reset_index()
        
        plt.figure(figsize=(20,8))
        plt.title('Forecasting day %d'%w, fontsize=20)
        plt.plot(data['Target'][-previous_steps:], lw=2.5, label='Actual', color='#03396c')
        plt.plot(temp[['%s'%models[0],'time']].set_index('time'), lw=2.5, label='%s'%legend[0])
        max_ = temp.drop(['Target','time'],axis=1).values.max()
        plt.vlines(x=temp['time'].min(),ymin=0, ymax=max_, linestyle='dashed', color='#4a4e4d', label='Start Forecasting')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Value', fontsize=18)
        plt.legend()
        plt.show()
        
    else:
        temp = Results['%s'%models[0]]['%s'%day_week]['%d'%w]['results'].rename(columns={'Predictions':'%s'%models[0]})\
               .reset_index()
        
        plt.figure(figsize=(20,8))
        plt.title('Forecasting day %d'%w, fontsize=20)
        plt.plot(real['Target'][-previous_steps:], lw=2.5, color='#03396c', alpha=0.7, label='Actual')
        plt.plot(temp[['%s'%models[0],'time']].set_index('time'), label='%s'%legend[0])
        
        for i, model in enumerate(models[1:]):
            data = Results['%s'%model]['%s'%'Days']['%d'%w]['results'].rename(columns={'Predictions':'%s'%model})['%s'%model]
            plt.plot(data, label='%s'%legend[i+1])

        max_ = temp.drop(['Target','time'],axis=1).values.max()
        plt.vlines(x=temp['time'].min(),ymin=0, ymax=max_, linestyle='dashed', color='#4a4e4d', label='Start Forecasting')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Value', fontsize=18)
        plt.legend()
        plt.show()