#Basic libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import yfinance
import warnings
import datetime

#feature extraction libraries
from tsextract.plots.eval import actualPred, get_lag_corr, scatter
from tsextract.feature_extraction.extract import build_features, build_features_forecast
from tsextract.domain.statistics import mean, median, std

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
from keras.layers import Dense, Dropout, Input, concatenate, Conv1D, MaxPooling1D, Flatten

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

#Statistics libraries
import scipy
from  scipy.stats import  boxcox
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA



class Create_Univariate_CNN:
    """
    This class creates and train a MLP model for univariate TS.
    ...
    Attributes
    ----------
    data : Pandas dataframe
    Data to fit the model with.
    """
    
    def __init__(self, training):
        
        self.data = training
        self.sequence = self.data.values
        self.index = self.data.index
        print('Your object is ceated!')

    
    # split a univariate sequence into samples
    @tf.autograph.experimental.do_not_convert
    def split_sequence(self, sequence, train_test=False):
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
        
        ret : bool
        If True, the functions returns the transformed features.
        Returns
        ----------
        Two numpy arrays(X, y).
        """
        #create our dataset
        dataset = tf.data.Dataset.from_tensor_slices(self.sequence)
        #apply window function to the dataset to convert the sequence to features and target
        dataset = dataset.window(self.w_size, shift=1, drop_remainder=True)
        #flatten the dataset to numpy array to start dealing with it
        dataset = dataset.flat_map(lambda window: window.batch(self.w_size))
        #shuffling and splitting the dataset into X,y
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))
 
        self.X = np.array([x.numpy() for x, _ in dataset])
        self.y = np.array([y.numpy() for _, y in dataset]).flatten() 
        
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        self.n_features =  1 #len(self.data.columns)
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], self.n_features))
        
        if train_test:
            # Split data using train proportion of (70% - 30%)
            self.split_point = int(0.7 * len(self.sequence)) 
            self.X_train, self.X_test = self.X[:self.split_point], self.X[self.split_point:]
            self.y_train, self.y_test = self.y[:self.split_point], self.y[self.split_point:]
            return self.X_train, self.X_test, self.y_train, self.y_test
                
        if self.is_scale:
            self.X, self.y = self.scale_data(self.X, self.y) 
            
        return self.X, self.y
     
    def scale_data(self, X, y):

        self.scaler_features = MinMaxScaler().fit(X.reshape(-1, 1))
        self.scaled_features = self.scaler_features.transform(X.reshape(-1, 1)).reshape(np.shape(X))

        self.scaler_label = StandardScaler().fit(np.array(y.reshape(-1, 1)))
        self.scaled_label = self.scaler_label.transform(y.reshape(-1, 1)).reshape(np.shape(y))
        
        return self.scaled_features, self.scaled_label
    
    def build_cnn(self, w_size, filter_num=64, kernel_size=2, pool_size=2, scale=True,
                  units=100, epochs=100, lr=0.03, batch = 256):
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
        self.units = units
        self.batch = batch
        self.epochs = epochs
        self.is_scale = scale
        self.w_size = w_size
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        callback = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,
                                                 patience=5, verbose=0, mode='auto')
        # split the sequence into samples
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_sequence(self.sequence, train_test=True)
        
        #define the optimizer
        self.model = Sequential()
        self.model.add(Conv1D(filters=self.filter_num, kernel_size=self.kernel_size, kernel_initializer='normal', 
                              activation='relu', input_shape=(self.w_size-1,#X.shape[0]
                                                              self.n_features)))
        self.model.add(MaxPooling1D(pool_size=self.pool_size))
        self.model.add(Flatten())
        self.model.add(Dense(self.units, activation='relu'))
        self.model.add(Dense(1))
        
        # Compile model
        opt = Adam(learning_rate=self.lr)
        self.model.compile(optimizer=opt, loss='mse')

        self.history = self.model.fit(x=self.X_train, y=self.y_train, 
                                      validation_data=(self.X_test, self.y_test),
                                      batch_size=self.batch,
                                      callbacks=callback,
                                      epochs=self.epochs)
    
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
        
    def plot_results(self):
        
        self.X, self.y = self.split_sequence(self.sequence)
        
        if self.is_scale:
            pred = self.model.predict(self.X)
            pred = self.scaler_label.inverse_transform(pred)
            actual = self.scaler_label.inverse_transform(self.y)
            
        else:
            pred = self.model.predict(self.X)
            actual = self.y
            
        temp = pd.DataFrame({'Actual':actual.flatten(), 'Forecasting':pred.flatten()})    
        
        temp.index = self.index[self.w_size-1:][-len(temp):] #lost data
        
        fig = go.Figure()
        fig['layout'] = dict(title='Forcasting Train-Test data',
                             titlefont=dict(size=20),
                             xaxis=dict(title='Date', titlefont=dict(size=18)),
                             yaxis=dict(title='Value', titlefont=dict(size=18),))
        
        fig.add_scatter(x=temp.index, y=temp['Actual'], name='Actual')
        fig.add_vline(x=temp.index[self.split_point], line_width=3, line_dash='dash', line_color='black')
        fig.add_scatter(x=temp.index, y=temp['Forecasting'], name='Forecasting')
        
        fig.show()
        
    def forecast(self, data):
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
        self.sequence = data.values
        self.test_index = data.index[self.w_size-1:] #w_size - 1 is lost data
        
#         n = min(n, len(self.sequence))
        #take the date to start forecasting from
        while True:
            start_forecasting = str(input('Enter the date to start forecasting from ({} to {})\n'.\
                                          format(self.test_index.min().strftime('%Y-%m-%d'), 
                                                 self.test_index.max().strftime('%Y-%m-%d'))))

            if start_forecasting not in self.test_index:
                print('Date is not in range of data')
            else:
                break

        start_forecasting_index = data.index.get_loc(start_forecasting)
        
        n = len(self.sequence[start_forecasting_index:])
        
        self.test_X, self.test_y = self.split_sequence(self.sequence)
        
        if self.is_scale:
            pred = self.model.predict(self.test_X)
            pred = self.scaler_label.inverse_transform(pred)
            actual = self.scaler_label.inverse_transform(self.test_y)
            
        else:
            pred = self.model.predict(self.X)
            actual = self.test_y
            
        temp = pd.DataFrame({'Actual':actual.flatten() ,'Forecasting':pred.flatten()})
        self.test_index = self.test_index[self.w_size-1:][-len(temp):] #lost data
        start = max(0, -200-n)

        fig = go.Figure()
        fig['layout'] = dict(title='Forcasting From {} to {}\n'.\
                             format(start_forecasting,
                                    self.test_index[-1].strftime('%Y-%m-%d')),
                             titlefont=dict(size=20),
                             xaxis=dict(title='Date', titlefont=dict(size=18)),
                             yaxis=dict(title='Value', titlefont=dict(size=18),))
        fig.add_scatter(x=self.test_index[start:], y=temp.iloc[start:]['Actual'], name='Actual')
        fig.add_vline(x=start_forecasting, line_width=3, line_dash='dash', line_color='black')
        fig.add_scatter(x=self.test_index[start:], y=temp.iloc[start:]['Forecasting'], name='Forecasting')
        fig.show()
        
        
class Create_Multivariate_Multple_Input:
    """
    This class creates and train a MLP model for multivariate TS.
    ...
    Attributes
    ----------
    data : Pandas dataframe
    Data to fit the model with.
    methode :  str
    Methode for modeling('single cnn' or 'multi headed')
    Single dense is a MLP model with only one MLP.
    Multi dense is a MLP model with one dense layer for each submodel.
    """
    def __init__(self, training, methode):
        
        self.methode = re.sub(r'[^\w]', ' ', methode).lower()
        
        if self.methode not in ['single cnn', 'multi headed']:
            raise ValueError('Invalid {}. Expecting single cnn or multi headed.'.format(self.methode))
        
        self.sequence = training.values
        self.index = training.index
        self.num_cols = len(training.columns)
        self.n_features = self.num_cols - 1
        
        print('Your Multivarite object is ceated!')

    # split a univariate sequence into samples
    @tf.autograph.experimental.do_not_convert
    def split_sequences(self, data, train_test=False):
        '''
        Docstring:
        This function transforms the raw dataset(TS) into input(Features) and output.
        ...
        Attributes
        ----------
        data : Numpy array
        The data you want to transform.
        w_size : integar number
        The size of applied window.
        
        Returns
        ----------
        Two numpy arrays(X, y).
        '''
        #create our dataset
        dataset = tf.data.Dataset.from_tensor_slices(data)
        #apply window function to the dataset to convert the sequence to features and target
        dataset = dataset.window(self.w_size, shift=1, drop_remainder=True)
        #flatten the dataset to numpy array to start dealing with it
        dataset = dataset.flat_map(lambda window: window.batch(self.w_size))
        # splitting the dataset into X,y
        if self.methode=='single cnn':
            dataset = dataset.map(lambda window: (window[:-1,:-1], window[-1:,-1:][0]))
            self.X = np.array([x.numpy() for x, _ in dataset])
            self.y = np.array([y.numpy() for _, y in dataset]).flatten()
            self.n_input = self.X.shape[1] * self.X.shape[2]
            # reshape from [samples, timesteps] into [samples, timesteps, features]
            self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], self.n_features))
        
        else:
            dataset = dataset.map(lambda window: (window[:-1], window[-2:][0][-1]))#if concate(feat_n,out)==>take right_button
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
        self.scaled_features = self.scaler_features.transform(X.reshape(-1, 1))\
                                                   .reshape(np.shape(X))

        self.scaler_label = StandardScaler().fit(np.array(y.reshape(-1, 1)))
        self.scaled_label = self.scaler_label.transform(y.reshape(-1, 1))
        
        return self.scaled_features, self.scaled_label
  
    def build_cnn(self, w_size, filter_num=64, kernel_size=2, pool_size=2, 
                  units=100, epochs=100, lr=0.03, batch = 256, scale=False):
        """
        This function creates and train a model for Multivarite TS.
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
        self.units = units
        self.batch = batch
        self.epochs = epochs
        self.w_size = w_size
        self.is_scale=scale
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        
        callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0,
                                                 patience=5,
                                                 verbose=0, 
                                                 mode='auto')
        if self.methode=='single cnn':
            # split the sequence into samples
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_sequences(self.sequence, train_test=True)

            self.model = Sequential()
            self.model.add(Conv1D(filters=self.filter_num, kernel_size=self.kernel_size, kernel_initializer='normal', 
                                  activation='relu', input_shape=(self.w_size-1, self.n_features)))
            self.model.add(MaxPooling1D(pool_size=self.pool_size))
            self.model.add(Flatten())
            self.model.add(Dense(self.units, activation='relu'))
            self.model.add(Dense(1))

            # Compile model
            opt = Adam(learning_rate=self.lr)
            self.model.compile(optimizer=opt, loss='mse')

            self.history = self.model.fit(x=self.X_train, y=self.y_train, 
                                          validation_data=(self.X_test, self.y_test),
                                          batch_size=self.batch, 
                                          callbacks=callback,
                                          epochs=self.epochs)

            return self.model

        else:
            # Data Prepration
            # convert into input/output
            self.X, self.y = self.split_sequences(self.sequence, train_test=False)
            
            self.n_features = 1
            
            # separate input data
            self.split_point = int(0.7 * np.shape(self.X[:, :, 0])[0])

            self.data = {}
            for i in range(self.num_cols-1):
                self.data['X_%d'%i] = {}
                self.data['X_%d'%i]['train'] = self.X[:, :, i].reshape(self.X.shape[0],self.X.shape[1],self.n_features)[:self.split_point]
                self.data['X_%d'%i]['test'] =  self.X[:, :, i].reshape(self.X.shape[0],self.X.shape[1],self.n_features)[self.split_point:]

            self.y_train, self.y_test = self.y[:self.split_point], self.y[self.split_point:]
        
            # build the submodels(model for each feature)
            self.model_layers = {'inputs':{},'convs':{},'pools':{},'flattens':{},'dense':{}}
            for i in range(self.num_cols-1):
                # input layer
                input_layer = Input(shape=(self.w_size-1, self.n_features))
                self.model_layers['inputs']['input_%s'%i] = input_layer
                # cnn layer
                cnn_layer = Conv1D(filters=self.filter_num, kernel_size=self.kernel_size, kernel_initializer='normal', 
                                   activation='relu', input_shape=(self.w_size-1,self.n_features))
                self.model_layers['convs']['cnn_%s'%i] = cnn_layer(self.model_layers['inputs']['input_%s'%i])
                #pooling layer
                pool_layer = MaxPooling1D(pool_size=self.pool_size, padding='same')
                self.model_layers['pools']['pool_%s'%i] = pool_layer(self.model_layers['convs']['cnn_%s'%i])
                #flatten layer
                flatten_layer = Flatten()
                self.model_layers['flattens']['flatten_%s'%i] = flatten_layer(self.model_layers['pools']['pool_%s'%i])
                #dense layer
                dense_layer = Dense(16, activation='tanh')
                self.model_layers['dense']['dense_%s'%i] = dense_layer(self.model_layers['flattens']['flatten_%s'%i])                
            
            # merge input models
            merge = concatenate([dense for dense in self.model_layers['dense'].values()])
            output = Dense(self.units, activation='relu')(merge)
            output = Dense(1)(output)
            
            # connect input and output models
            self.model = Model(inputs=[inp for inp in self.model_layers['inputs'].values()], outputs=output)
            
            # Compile the model
            opt = Adam(learning_rate=self.lr)
            self.model.compile(optimizer=opt, loss='mse')
            
            # fit the model
            self.history = self.model.fit([self.data['X_%d'%col]['train'] for col in range(self.num_cols-1)], self.y_train, 
                                          callbacks=callback,
                                          epochs=self.epochs, 
                                          verbose=1, 
                                          #batch_size=self.batch,
                                          validation_data=([[self.data['X_%d'%col]['test'] for col in range(self.num_cols-1)]],
                                                           [self.y_test]))
    
            return self.model
    
    def show_train_test_results(self):
        
        temp = pd.DataFrame({'Training':self.history.history['loss'],
                             'Validation':self.history.history['val_loss']})

        fig = px.line(temp)
        fig.show()
        
    def plot_results(self):     
        
        if self.methode=='single cnn':
            if self.is_scale:
                pred_data = self.model.predict(self.X)
                pred_data = self.scaler_label.inverse_transform(pred_data)
                actual_data = self.scaler_label.inverse_transform(self.y)
                
            else:
                pred_data = self.model.predict(self.X)
                actual_data = self.y
                
        else:
            X_, actual_data = self.split_sequences(self.sequence)
            X_ = [self.X[:, :, i] for i in range(self.num_cols-1)]
            
            pred_data = self.model.predict(X_)            

        temp = pd.DataFrame({'Actual':actual_data.flatten(), 'Forecasting':pred_data.flatten()})    
        temp.index = self.index[-len(temp):]

        fig = go.Figure()
        fig['layout'] = dict(title='Forcasting Train-Test data',
                             titlefont=dict(size=20),
                             xaxis=dict(title='Date', titlefont=dict(size=18)),
                             yaxis=dict(title='Value', titlefont=dict(size=18),))
        
        fig.add_scatter(x=temp.index, y=temp['Actual'], name='Actual')
        fig.add_vline(x=temp.index[self.split_point], line_width=3, line_dash='dash', line_color='black')
        fig.add_scatter(x=temp.index, y=temp['Forecasting'], name='Forecasting')
        
        fig.show()
        
        
    def forecast(self, data):
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
        self.sequence = data.values
        self.test_index = data.index[self.w_size-1:] # self.w_size-1 lost data
        
        #take the date to start forecasting from
        while True:
            start_forecasting = str(input('Enter the date to start forecasting from ({} to {})\n'.\
                                          format(self.test_index.min().strftime('%Y-%m-%d'), 
                                                 self.test_index.max().strftime('%Y-%m-%d'))))

            if start_forecasting not in self.test_index:
                print('Date is not in range of data')
            else:
                break

        start_forecasting_index = data.index.get_loc(start_forecasting)
        
        n = len(self.sequence[start_forecasting_index:])

        start = max(0, -200-n)

        self.test_X, self.test_y = self.split_sequences(self.sequence, train_test=False)
        
        if self.methode!='single cnn':
            self.test_X = [self.test_X[:, :, i] for i in range(self.num_cols-1)]
        
        if self.is_scale:
            pred = self.model.predict(self.test_X)
            pred = self.scaler_label.inverse_transform(pred)
            self.test_y = self.scaler_label.inverse_transform(self.test_y)
        else:
            pred = self.model.predict(self.test_X)

        temp = pd.DataFrame({'Actual':self.test_y.flatten() ,'Forecasting':pred.flatten()})
        
        fig = go.Figure()
        fig['layout'] = dict(title='Forcasting From {} to {}'.\
                             format(start_forecasting,
                                    self.test_index[-1].strftime('%Y-%m-%d')),
                             titlefont=dict(size=20),
                             xaxis=dict(title='Date', titlefont=dict(size=18)),
                             yaxis=dict(title='Value', titlefont=dict(size=18),))
        fig.add_scatter(x=self.test_index[start:], y=temp.iloc[start:]['Actual'], name='Actual')
        fig.add_vline(x=start_forecasting, line_width=3, line_dash='dash', line_color='black')
        fig.add_scatter(x=self.test_index[start:], y=temp.iloc[start:]['Forecasting'], name='Forecasting')
        fig.show()
        
        
class Create_Multivariate_Multple_Parallel_:
    """
    This class creates and train a MLP model for multivariate TS.
    ...
    Attributes
    ----------
    data : Pandas dataframe
    Data to fit the model with.
    methode :  str
    Methode for modeling('vector output' or 'multi output')
    """
    def __init__(self, training, methode):
        
        self.methode = re.sub(r'[^\w]', ' ', methode).lower()
        
        if self.methode not in ['vector output', 'multi output']:
            raise ValueError('Invalid {}. Expecting vector output or multi output.'.format(self.methode))
        
        self.sequence = training.values
        self.index = training.index
        self.columns = [col.capitalize() for col in training.columns]
        self.num_cols = len(training.columns)
        
        print('Your Multivarite object is ceated!')
        
    # split a multivariate sequence into samples
    @tf.autograph.experimental.do_not_convert
    def split_sequences(self, sequence):
        
        #create our dataset
        dataset = tf.data.Dataset.from_tensor_slices(sequence)
        # apply window function to the dataset to convert the sequence to features and target
        dataset = dataset.window(self.w_size, shift=1, drop_remainder=True)
        # flatten the dataset to numpy array to start dealing with it
        dataset = dataset.flat_map(lambda window: window.batch(self.w_size))
        # #shuffling and splitting the dataset into X,y
        dataset = dataset.map(lambda window: (window[:-1,:], window[-1:,:][0]))

        self.X = np.array([x.numpy() for x, _ in dataset])
        self.y = np.array([y.numpy() for _, y in dataset])
        
        # determine the number of outputs
        self.n_features = self.X.shape[2]
            
        # flatten input
        self.n_features = self.X.shape[2]
#         self.X = self.X.reshape((self.X.shape[0], self.n_input))
        
        # Scaling the data
        if self.is_scale:
            self.X, self.y = self.scale_data(self.X, self.y)
            
        ### Split data using train proportion of (70% - 30%)
        self.split_point = int(0.7 * len(self.X)) 
        self.X_train, self.X_test = self.X[:self.split_point], self.X[self.split_point:]
        self.y_train, self.y_test = self.y[:self.split_point], self.y[self.split_point:]
        self.train_index, self.test_index = self.index[:self.split_point], self.index[self.split_point:]  

        return self.X, self.y


    def scale_data(self, X, y):

        self.scaler_features = MinMaxScaler().fit(X.reshape(-1, 1))
        self.scaled_features = self.scaler_features.transform(X.reshape(-1, 1)).reshape(np.shape(X))

        self.scaler_label = StandardScaler().fit(np.array(y.reshape(-1, 1)))
        self.scaled_label = self.scaler_label.transform(y.reshape(-1, 1)).reshape(np.shape(y))
        
        return self.scaled_features, self.scaled_label
        
    def build_cnn(self, w_size, filter_num=64, kernel_size=2, pool_size=2, 
                  units=100, epochs=100, lr=0.03, batch = 256, scale=False):
        
        self.lr = lr
        self.units = units
        self.batch = batch
        self.epochs = epochs
        self.w_size = w_size
        self.is_scale = scale
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        callback = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,
                                                 patience=5, verbose=0, mode='auto')
        
        # split the sequence into samples
        self.X, self.y = self.split_sequences(self.sequence)
        
        # determine the number of outputs
        self.n_features = self.X.shape[2]
        
        # Scaling the data
        if self.is_scale:
            self.X, self.y = self.scale_data(self.X, self.y)
            
        # flatten input
        self.n_features = self.X.shape[2]
#         self.X = self.X.reshape((self.X.shape[0], self.n_input))
        
        ### Split data using train proportion of (70% - 30%)
        self.split_point = int(0.7 * len(self.X)) 
        self.X_train, self.X_test = self.X[:self.split_point], self.X[self.split_point:]
        self.y_train, self.y_test = self.y[:self.split_point], self.y[self.split_point:]
        self.train_index, self.test_index = self.index[:self.split_point], self.index[self.split_point:]                  
        
        if self.methode=='vector output':    
            
            # Build the model        
            opt = Adam(learning_rate=self.lr)
            self.model = Sequential()
            self.model.add(Conv1D(filters=self.filter_num, kernel_size=self.kernel_size, kernel_initializer='normal', 
                                  activation='relu', input_shape=(self.w_size-1, self.n_features)))
            self.model.add(MaxPooling1D(pool_size=self.pool_size))
            self.model.add(Flatten())
            self.model.add(Dense(self.units, activation='relu'))
            self.model.add(Dense(self.n_features))

            # Compile model
            opt = Adam(learning_rate=self.lr)
            self.model.compile(optimizer=opt, loss='mse')

            self.history = self.model.fit(x=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test),
#                                           batch_size=self.batch,
                                          callbacks=callback,
                                          epochs=self.epochs)

        
        else:
            
            # Build the model        
            opt = Adam(learning_rate=self.lr)
            input_ = Input(shape=(self.w_size-1, self.n_features))
            cnn = Conv1D(filters=self.filter_num, kernel_size=self.kernel_size, kernel_initializer='normal', 
                                  activation='relu', input_shape=(self.w_size-1, self.n_features))(input_)
            cnn = MaxPooling1D(pool_size=self.pool_size)(cnn)
            cnn = Flatten()(cnn)
            cnn = Dense(self.units, activation='relu')(cnn)
            
            # split the output(for each output layer)
            self.outputs = {'output_%d'%i:{} for i in range(self.n_features)}
            
            # create output layer each for one TS
            self.output_layers = {'output_%d'%i:None for i in range(self.n_features)}
            
            for i in range(self.n_features):
                self.outputs['output_%d'%i]['train'] = self.y[:, i].reshape((self.y.shape[0], 1))[:self.split_point]
                self.outputs['output_%d'%i]['test'] = self.y[:, i].reshape((self.y.shape[0], 1))[self.split_point:]
                self.output_layers['output_%d'%i] = Dense(1)(cnn)
            
            # tie together
            self.model = Model(inputs=input_, outputs=[out for out in self.output_layers.values()])

            # Compile model
            opt = Adam(learning_rate=self.lr)
            self.model.compile(loss='mean_squared_error', optimizer=opt)

            self.history = self.model.fit(x=self.X_train, 
                                          y=[self.outputs['output_%d'%i]['train'] for i in range(self.n_features)],
                                          validation_data=(self.X_test, 
                                                           [self.outputs['output_%d'%i]['test'] for i in range(self.n_features)]),
                                          batch_size=self.batch, 
                                          callbacks=callback,
                                          epochs=self.epochs)

    def show_train_test_results(self):
        
        if self.methode=='vector output':
            temp = pd.DataFrame({'Training':self.history.history['loss'],
                             'Validation':self.history.history['val_loss']})

            fig = px.line(temp)
            fig.show()
        
        
        else:
            columns = self.columns
            num_cols = self.num_cols

            temp = pd.DataFrame(self.history.history)
            #rename the columns of training losses for all TS.
            temp = temp.rename(columns={temp.columns[1:][i] : columns[i]+'_loss' for i in range(4)})

            #rename the columns for validation losses
            temp = temp.rename(columns={temp.columns[num_cols+2:][i] : columns[i]+'_val_loss' for i in range(len(columns[1:]))})

            #rename the first layer columns of training and validation losses 
            temp = temp.rename(columns={'loss':'First_dennse_loss'})
            temp = temp.rename(columns={'val_loss':'First_dennse_val_loss'})

            fig = px.line(temp)
            fig.show()
        
    
    def plot_results(self):
        
        if self.methode=='vector output':
            
            self.X, self.y = self.scale_data(self.X, self.y)
            
            if self.is_scale:
                pred = self.model.predict(self.X)
                pred = self.scaler_label.inverse_transform(pred)
                actual = self.scaler_label.inverse_transform(self.y)
                
            else:
                pred = self.model.predict(self.X)
                actual = self.y
                
            for i in range(self.num_cols):
                temp = pd.DataFrame({'Actual':self.y[:,i].flatten(), 'Forecasting':pred[:,i].flatten()})
                fig=px.line(temp)
                fig['layout'] = dict(title='Forcasting Train-Test data for {}'.format(self.columns[i]),
                                     titlefont=dict(size=20),
                                     xaxis=dict(title='Date', titlefont=dict(size=18)),
                                     yaxis=dict(title='Value', titlefont=dict(size=18)))
                fig.add_vline(x=temp.index[self.split_point], line_width=3, line_dash='dash', line_color='black')
                fig.show()
                
        else:
            
            if self.is_scale:
                pred_data = self.model.predict(self.X)
                pred_data = self.scaler_label.inverse_transform(pred_data)
                actual_data = self.scaler_label.inverse_transform(self.y)
                
            else:
                pred_data = self.model.predict(self.X)
                actual_data = self.y
                
            for i in range(self.n_features):
                temp = pd.DataFrame({'Actual':actual_data[:,i].flatten(),
                                     'Predictions':pred_data[i].flatten()})
                temp.index = self.index[:len(temp)]
                fig = px.line(temp)
                fig['layout']=dict(title=self.columns[i], titlefont=dict(size=18))
                fig.add_vline(x=temp.index[self.split_point], line_width=3, line_dash='dash', line_color='black')
                fig.show()


    def forecast(self, data):
         
        self.data = data.values
        self.test_index = data.index[self.w_size-1:]

        #take the date to start forecasting from
        while True:
            start_forecasting = str(input('Enter the date to start forecasting from ({} to {})\n'.\
                                          format(self.test_index.min().strftime('%Y-%m-%d'), 
                                                 self.test_index.max().strftime('%Y-%m-%d'))))

            if start_forecasting not in self.test_index:
                print('Date is not in range of data')
            else:
                break

        start_forecasting_index = data.index.get_loc(start_forecasting)
        
        n = len(self.sequence[start_forecasting_index:])
        start = max(0, -200-n)
        
        self.test_X, self.test_y = self.split_sequences(self.data)
        
        if self.methode=='vector output':
            if self.is_scale:
                pred = self.model.predict(self.test_X)
                pred = self.scaler_label.inverse_transform(pred)
                actual = self.scaler_label.inverse_transform(self.test_y)

            else:
                pred = self.model.predict(self.test_X)
                actual = self.test_y
            
            for i in range(self.num_cols):
                temp = pd.DataFrame({'Actual': actual[:,i].flatten(), 'Forecasting': pred[:,i].flatten()})
                temp.index = self.test_index[-len(temp):]
                
                fig = go.Figure()
                fig['layout'] = dict(title='Forcasting {} From {} to {}'.\
                                     format(self.columns[i],
                                            start_forecasting,
                                            self.test_index[-1].strftime('%Y-%m-%d')),
                                     titlefont=dict(size=20),
                                     xaxis=dict(title='Date', titlefont=dict(size=18)),
                                     yaxis=dict(title='Value', titlefont=dict(size=18),))
                fig.add_scatter(x=temp.index[start:], y=temp.iloc[start:]['Actual'], name='Actual')
                fig.add_vline(x=start_forecasting, line_width=3, line_dash='dash', line_color='black')
                fig.add_scatter(x=temp.index[start:], y=temp.iloc[start:]['Forecasting'], name='Forecasting')
                fig.show()
                               
        else:
            # split the sequence into samples
            self.X, self.y = self.split_sequences(self.sequence)
            
            if self.is_scale:
                pred_data = self.model.predict(self.X)
                pred_data = self.scaler_label.inverse_transform(pred_data)
                actual_data = [self.y[:, i].reshape((self.y.shape[0], 1)) for i in range(self.n_features)]
                actual_data = self.scaler_label.inverse_transform(actual_data)

            else:
                actual_data = [self.y[:, i].reshape((self.y.shape[0], 1)) for i in range(self.n_features)]
                pred_data = self.model.predict(self.X)
            
            
            
            
            for i in range(self.n_features):
                
                temp = pd.DataFrame({'Actual':actual_data[i].flatten() ,
                                     'Forecasting':pred_data[i].flatten()})

                fig = go.Figure()
                fig['layout'] = dict(title='Forcasting {} From {} to {}'.\
                                     format(self.columns[i],
                                            start_forecasting,
                                            self.test_index[-1].strftime('%Y-%m-%d')),
                                     titlefont=dict(size=20),
                                     xaxis=dict(title='Date', titlefont=dict(size=18)),
                                     yaxis=dict(title='Value', titlefont=dict(size=18),))
                fig.add_scatter(x=self.test_index[start:], y=temp.iloc[start:]['Actual'], name='Actual')
                fig.add_vline(x=start_forecasting, line_width=3, line_dash='dash', line_color='black')
                fig.add_scatter(x=self.test_index[start:], y=temp.iloc[start:]['Forecasting'], name='Forecasting')
                fig.show()
                
                
class Create_Univariate_Multi_Step_CNN:
    """
    This class creates and train a MLP model for univariate TS.
    ...
    Attributes
    ----------
    data : Pandas dataframe
    Data to fit the model with.
    """
    
    def __init__(self, training):
        
        self.sequence = training.values
        self.index = training.index
        print('Your object is ceated!')

    # split a univariate sequence into samples
    @tf.autograph.experimental.do_not_convert
    def split_sequence(self, data):

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
        
        ret : bool
        If True, the functions returns the transformed features.
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
        
        return self.X, self.y
      
    def scale_data(self, X, y):

        self.scaler_features = MinMaxScaler().fit(X.reshape(-1, 1))
        self.scaled_features = self.scaler_features.transform(X.reshape(-1, 1)).reshape(np.shape(X))

        self.scaler_label = StandardScaler().fit(np.array(y.reshape(-1, 1)))
        self.scaled_label = self.scaler_label.transform(y.reshape(-1, 1)).reshape(np.shape(y))
        
        return self.scaled_features, self.scaled_label
        
    def build_cnn(self, n_steps_in, n_steps_out, filter_num=64, kernel_size=2, pool_size=2,
                  units=100, epochs=100, lr=0.03, batch = 256, scale=False):
        
        self.lr = lr
        self.units = units
        self.batch = batch
        self.epochs = epochs
        self.is_scale = scale
        self.n_steps_in =  n_steps_in
        self.n_steps_out = n_steps_out
        self.w_size = self.n_steps_in + self.n_steps_out
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        callback = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,
                                                 patience=5, verbose=0, mode='auto')
        
        # split the sequence into samples
        self.X, self.y = self.split_sequence(self.sequence)

        if self.is_scale:
            self.X, self.y = self.scale_data(self.X, self.y)
            
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        self.n_features = 1
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], self.n_features))
        
        # Split data using train proportion of (70% - 30%)
        self.split_point = int(0.7 * len(self.X)) 
        self.X_train, self.X_test = self.X[:self.split_point], self.X[self.split_point:]
        self.y_train, self.y_test = self.y[:self.split_point], self.y[self.split_point:]
        
        #define the optimizer
        opt = Adam(learning_rate=self.lr)

        # Build the model
        self.model = Sequential()
        self.model.add(Conv1D(filters=self.filter_num, kernel_size=self.kernel_size, kernel_initializer='normal', 
                              activation='relu', input_shape=(self.n_steps_in, self.n_features)))
        self.model.add(MaxPooling1D(pool_size=self.pool_size))
        self.model.add(Flatten())
        self.model.add(Dense(self.units, activation='relu'))
        self.model.add(Dense(self.n_steps_out))
        
        # Compile model
        opt = Adam(learning_rate=self.lr)
        self.model.compile(optimizer=opt, loss='mse')

        self.history = self.model.fit(x=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test),
#                                       batch_size=self.batch, 
                                      callbacks=callback,
                                      epochs=self.epochs)

    
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

    def forecast(self, data):
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
        self.data = data
        self.test_data = data.values
        self.test_index = data.index
        
        #take the date to start forecasting from
        while True:
            start_forecasting = str(input('Enter the date to start forecasting from ({} to {})'.\
                                          format(self.test_index.min().strftime('%Y-%m-%d'), 
                                                 self.test_index.max().strftime('%Y-%m-%d'))))

            if start_forecasting not in self.test_index:
                print('Date is not in range of data')
            if start_forecasting.lower()=='end':
                return False
            else:
                break

        start_forecasting_index = data.index.get_loc(start_forecasting)

        self.test_X, self.test_y = self.split_sequence(self.sequence)
        
        n = len(self.test_data[start_forecasting_index:])
        
        start_forecasting_index = self.data.index.get_loc(start_forecasting)
        
        self.forecast_data = self.data[start_forecasting_index-self.n_steps_in:start_forecasting_index+self.n_steps_out]

        self.test_X, self.test_y = self.forecast_data[:self.n_steps_in], self.forecast_data[self.n_steps_in:]

        pred = self.model.predict([self.test_X.values.reshape((1, self.n_steps_in))]).flatten()
        
        pred = pd.Series(pred, index=self.forecast_data.index[self.n_steps_in:])

        start = abs(self.data.index.get_loc(start_forecasting) - self.n_steps_in)
        start = self.data.reset_index().loc[start]['index'].strftime('%Y-%m-%d')
        
        end = self.data.index.get_loc(start_forecasting) + self.n_steps_out
        end = self.data.reset_index().loc[end]['index'].strftime('%Y-%m-%d')
        
        fig = go.Figure()
        
        fig['layout'] = dict(title='Forcasting From {} to {}'.format(start, end),
                             titlefont=dict(size=20),
                             xaxis=dict(title='Date', titlefont=dict(size=18)),
                             yaxis=dict(title='Value', titlefont=dict(size=18),))
        
        fig.add_scatter(x=self.forecast_data.index, y=self.forecast_data.values, name='Actual')
        
        fig.add_vline(x=self.forecast_data.index[self.n_steps_in], line_width=3, line_dash='dash', line_color='black')
        
        fig.add_scatter(x=pred.index, y=pred.values, name='Forecasting')
        
        fig.show()
        
        
class Create_Multivariate_Multi_Step:
    """
    This class creates and train a MLP model for multivariate TS.
    ...
    Attributes
    ----------
    data : Pandas dataframe
    Data to fit the model with.
    methode :  str
    Methode for modeling('multiple input' or 'multiple parallel')
    Single dense is a MLP model with only one MLP.
    Multi dense is a MLP model with one dense layer for each submodel.
    """
    def __init__(self, training, methode):
        
        self.methode = re.sub(r'[^\w]', ' ', methode).lower()
        
        if self.methode not in ['multiple input', 'multiple parallel']:
            raise ValueError('Invalid {}. Expecting vector output or multi output.'.format(self.methode))
        
        self.sequence = training.values
        self.index = training.index
        self.columns = [col[0].capitalize() for col in training.columns]
        self.num_cols = len(training.columns)
        
        print('Your Multivarite object is ceated!')
        
    # split a multivariate sequence into samples
    @tf.autograph.experimental.do_not_convert
    def split_sequences(self, sequence, train_test=False):
        
        #create our dataset
        dataset = tf.data.Dataset.from_tensor_slices(sequence)
        # apply window function to the dataset to convert the sequence to features and target
        dataset = dataset.window(self.w_size, shift=1, drop_remainder=True)
        # flatten the dataset to numpy array to start dealing with it
        dataset = dataset.flat_map(lambda window: window.batch(self.w_size))
        # #shuffling and splitting the dataset into X,y
        if self.methode=='multiple input':
            dataset = dataset.map(lambda window: (window[:self.n_steps_in,:-1], window[self.n_steps_in-1:self.w_size-1,-1:]))
            self.X = np.array([x.numpy() for x, _ in dataset])
            self.y = np.array([y.numpy() for _, y in dataset])
            # determine the number of outputs
            self.n_input = self.X.shape[1] * self.X.shape[2]   
            # determine the number of outputs
            self.n_output = self.y.shape[1] 
            # Scaling the data
            if self.is_scale:
                self.X, self.y = self.scale_data(self.X, self.y)
                
        else:
            dataset = dataset.map(lambda window: (window[:self.n_steps_in,:], window[self.n_steps_in:self.w_size,:]))
            self.X = np.array([x.numpy() for x, _ in dataset])
            self.y = np.array([y.numpy() for _, y in dataset])
            # Scaling the data
            if self.is_scale:
                self.X, self.y = self.scale_data(self.X, self.y)
            # flatten output
            self.n_output = self.y.shape[1] * self.y.shape[2]
            self.y = self.y.reshape((self.y.shape[0], self.n_output))
            # the number of features
            self.n_features = self.X.shape[2]
        
        if train_test:
            ### Split data using train proportion of (70% - 30%)
            self.split_point = int(0.7 * len(self.X))
            self.X_train, self.X_test = self.X[:self.split_point], self.X[self.split_point:]
            self.y_train, self.y_test = self.y[:self.split_point], self.y[self.split_point:]
            self.train_index, self.test_index = self.index[:self.split_point], self.index[self.split_point:]
        
        return self.X, self.y

    def scale_data(self, X, y):

        self.scaler_features = MinMaxScaler().fit(X.reshape(-1, 1))
        self.scaled_features = self.scaler_features.transform(X.reshape(-1, 1)).reshape(np.shape(X))

        self.scaler_label = StandardScaler().fit(np.array(y.reshape(-1, 1)))
        self.scaled_label = self.scaler_label.transform(y.reshape(-1, 1)).reshape(np.shape(y))
        
        return self.scaled_features, self.scaled_label
    
    def build_cnn(self, n_steps_in, n_steps_out, filter_num=64, kernel_size=2, pool_size=2, 
                  units=100, epochs=100, lr=0.03, batch = 256, scale=True):
        
        self.lr = lr
        self.units = units
        self.batch = batch
        self.epochs = epochs
        self.is_scale = scale
        self.n_steps_in =  n_steps_in
        self.n_steps_out = n_steps_out
        self.w_size = self.n_steps_in + self.n_steps_out
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_features = self.num_cols - 1
        callback = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,
                                                 patience=5, verbose=0, mode='auto')
        # split the sequence into samples
        self.X, self.y = self.split_sequences(self.sequence, train_test=True)
        
        if self.methode=='multiple input':    
            
            #define the optimizer
            opt = Adam(learning_rate=self.lr)

            # Build the model
            self.model = Sequential()
            self.model.add(Conv1D(filters=self.filter_num, kernel_size=self.kernel_size, kernel_initializer='normal', 
                                  activation='relu', input_shape=(self.n_steps_in, self.n_features)))
            self.model.add(Conv1D(filters=16, kernel_size=2, kernel_initializer='normal', activation='relu'))
            self.model.add(MaxPooling1D(pool_size=self.pool_size))
            self.model.add(Flatten())
            self.model.add(Dense(self.units, activation='relu'))
            self.model.add(Dense(self.n_steps_out))

            # Compile model
            opt = Adam(learning_rate=self.lr)
            self.model.compile(optimizer=opt, loss='mse')

            self.history = self.model.fit(x=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test),
    #                                       batch_size=self.batch, 
                                          callbacks=callback,
                                          epochs=self.epochs)

            return self.model
        
        else:
           # Build the model        
            opt = Adam(learning_rate=self.lr)
            self.model = Sequential()
            self.model.add(Conv1D(filters=self.filter_num, kernel_size=self.kernel_size, kernel_initializer='normal', 
                                  activation='relu', input_shape=(self.n_steps_in, self.n_features)))
            self.model.add(Conv1D(filters=16, kernel_size=2, kernel_initializer='normal', activation='relu'))
            self.model.add(MaxPooling1D(pool_size=self.pool_size, padding='same'))
            self.model.add(Flatten())
            self.model.add(Dense(self.n_output))

            # Compile model
            self.model.compile(loss='mean_squared_error', optimizer=opt)

            self.history = self.model.fit(x=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test),
#                                           batch_size=self.batch,
                                          callbacks=callback,
                                          epochs=self.epochs)

            return self.model
            
    def show_train_test_results(self):
        
        if self.methode=='multiple input':
            temp = pd.DataFrame({'Training':self.history.history['loss'],
                             'Validation':self.history.history['val_loss']})

            fig = px.line(temp)
            fig.show()
        
        
        else:
            columns = self.columns
            num_cols = self.num_cols

            temp = pd.DataFrame(self.history.history)

            fig = px.line(temp)
            fig.show()
              
                
    def forecast(self, data):
        
        self.data = data
        self.test_index = data.index
        #take the date to start forecasting from
        while True:
            start_forecasting = str(input('Enter the date to start forecasting from ({} to {})'.\
                                          format(self.test_index.min().strftime('%Y-%m-%d'), 
                                                 self.test_index.max().strftime('%Y-%m-%d'))))

            if start_forecasting not in self.test_index:
                print('Date is not in range of data')
            else:
                break

        start_forecasting_index = data.index.get_loc(start_forecasting)
        
        n = len(self.sequence[start_forecasting_index:])
        
        
        start_forecasting_index = self.data.index.get_loc(start_forecasting)
        
        self.forecast_data = self.data[start_forecasting_index-self.n_steps_in:start_forecasting_index+self.n_steps_out]
        
        self.test_X, self.test_y = self.split_sequences(self.forecast_data)

        pred = self.model.predict(self.test_X).flatten()
        pred = self.scaler_label.inverse_transform(pred)
#         return pred
        start = abs(self.data.index.get_loc(start_forecasting) - self.n_steps_in)
        start = self.data.reset_index().loc[start]['index'].strftime('%Y-%m-%d')
        
        end = self.data.index.get_loc(start_forecasting) + self.n_steps_out
        end = self.data.reset_index().loc[end]['index'].strftime('%Y-%m-%d')
        
        fig = go.Figure()
#         return self.forecast/_data
        fig['layout'] = dict(title='Forcasting From {} to {}'.format(start, end),
                             titlefont=dict(size=20),
                             xaxis=dict(title='Date', titlefont=dict(size=18)),
                             yaxis=dict(title='Value', titlefont=dict(size=18),))
        
        fig.add_scatter(x=self.forecast_data.index, 
                        y=self.forecast_data.iloc[:,-1:].values.flatten(), name='Actual')
        
        fig.add_vline(x=self.forecast_data.index[self.n_steps_in], line_width=3, line_dash='dash', line_color='black')
        
        fig.add_scatter(x=self.forecast_data.index[self.n_steps_in:], 
                        y=[i for i in pred], name='Forecasting')
        

        fig.show()