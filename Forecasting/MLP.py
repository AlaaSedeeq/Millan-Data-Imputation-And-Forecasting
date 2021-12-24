#Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import warnings
import datetime

#statistics libraries
import scipy
from  scipy.stats import  boxcox
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

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


def plot_line(data, y, title='', range_x=None):
    fig = px.line(data, y=y, range_x=range_x)
    fig['layout'] = dict(title=dict(text=title,  font=dict(size=20), xanchor='auto'),
                         xaxis=dict(title='Date', titlefont=dict(size=18)),
                         yaxis=dict(title='Value', titlefont=dict(size=18)))
    fig.show()
    
    
def plot_hist(data, title='', range_x=None):
    fig = px.histogram(data, range_x=range_x)
    fig['layout'] = dict(title=dict(text=title.title(), font=dict(size=20), xanchor='auto'),
                         xaxis=dict(title='Date', titlefont=dict(size=18)),
                         yaxis=dict(title='Count', titlefont=dict(size=18)))
    fig.show()

    
def test_stationarity(df, series, title='', ret_values=None):
    
    # Determing rolling statistics
    rolmean = df[series].rolling(window = 12, center = False).mean().dropna() #Checkif our data has constant mean
    rolstd = df[series].rolling(window = 12, center = False).std().dropna()   #Checkif our data has constant variance
    
    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    print('===>Results of Dickey-Fuller Test for %s:\n' %(series))
    dftest = sts.adfuller(df[series].dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic',
                                  'p-value',
                                  '# Lags Used',
                                  'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    print(f'Result: The series is {"not " if dfoutput[1] > 0.05 else ""}stationary')
    
    # Plot rolling statistics:
    fig = go.Figure()
    fig.add_scatter(x=tuple(range(len(df[series]))), y=df[series].values, name='TS Data')
    fig.add_scatter(x=tuple(range(len(rolmean.values))), y=rolmean.values, name='Rolling Mean')
    fig.add_scatter(x=tuple(range(len(rolstd.values))), y=rolstd.values, name='Rolling std')
    fig['layout'] = dict(title=title.title(), titlefont=dict(size=20),
                         xaxis=dict(title='Range', titlefont=dict(size=18)),
                         yaxis=dict(title='Values', titlefont=dict(size=18)))
    fig.show()
    
    if ret_values:
        return dfoutput[1]
    
def create_corr_plot(series, series_name='', plot_pacf=False):
    corr_array = pacf(series.dropna(), alpha=0.05, method='ols', nlags=40) if plot_pacf\
    else acf(series.dropna(), alpha=0.05, fft=False, nlags=40) #nlags=10*np.log10(len(series))
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure()
    
    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][1:][x]), mode='lines', line_color='#3f3f3f', 
                     name='lag{}'.format(x)) for x in range(len(corr_array[0][1:]))]
    
    [fig.add_scatter(x=[i], y=[corr_array[0][1:][i]], mode='markers', marker_color='#1f77b4', 
                     marker_size=12,name='lag{}'.format(i)) for i in np.arange(len(corr_array[0][1:]))]
    
    fig.add_scatter(x=np.arange(len(corr_array[0][1:])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)',
                    name='Confidence interval')
    
    fig.add_scatter(x=np.arange(len(corr_array[0][1:])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
                    fill='tonexty', line_color='rgba(255,255,255,0)', name='Confidence interval')
    
    fig.add_hrect(y0=2/((len(series)**0.5)), y1=-2/((len(series)**0.5)), line_width=0, fillcolor="red", opacity=0.5)

    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,42])
    fig.update_yaxes(zerolinecolor='#000000')
    
    title='Partial Autocorrelation function (PACF) of {}'.format(series_name) if plot_pacf\
     else 'Autocorrelation function (ACF) of {}'.format(series_name)
    
    fig['layout']=dict(title=title,titlefont=dict(size=20), width=1050)
    
    fig.show()
    
    
class Create_Univariate_MLP:
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
    def split_sequence(self, sequence):
        """
        Docstring:
        This function transforms the raw dataset(TS) into inputs(Features) and output.
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
        dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
        
        # separate features and output
        self.X = np.array([x.numpy() for x, _ in dataset])
        self.y = np.array([y.numpy() for _, y in dataset])
        
        #scaling the data 
        if self.is_scale:
            self.X, self.y = self.scale_data(self.X, self.y)
        
        return self.X, self.y
    
    def scale_data(self, X, y):

        self.scaler_features = MinMaxScaler().fit(X.reshape(-1, 1))
        self.scaled_features = self.scaler_features.transform(X.reshape(-1, 1)).reshape(np.shape(X))

        self.scaler_label = StandardScaler().fit(np.array(y.reshape(-1, 1)))
        self.scaled_label = self.scaler_label.transform(y.reshape(-1, 1)).reshape(np.shape(y))
        
        return self.scaled_features, self.scaled_label
      
    def build_dense_layers(self, w_size, epochs=100, lr=0.03, batch = 256, scale=True):
        """
        This function creates and train a MLP model for Univarite TS.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
        The data you want to see information about
        
        Returns
        ----------
        A trained model fitted to the data
        """
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.w_size = w_size
        self.is_scale = scale
        
        # split the sequence into samples
        self.X, self.y = self.split_sequence(self.sequence)
        
        callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0,
                                                 patience=5,
                                                 verbose=0, 
                                                 mode='auto')
        
        # Split data using train proportion of (70% - 30%)
        self.split_point = int(0.7 * len(self.X)) 
        self.X_train, self.X_test = self.X[:self.split_point], self.X[self.split_point:]
        self.y_train, self.y_test = self.y[:self.split_point], self.y[self.split_point:]
    
        # Build the model
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=(self.w_size-1), activation='relu'))
        self.model.add(Dense(1))

        #define the optimizer
        opt = Adam(learning_rate=self.lr)

        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer=opt)

        self.history = self.model.fit(x=self.X_train, y=self.y_train, 
                                      validation_data=(self.X_test, self.y_test),
                                      callbacks=callback,
                                      batch_size=self.batch, 
                                      epochs=self.epochs)

        return self.model
    
    def show_train_test_results(self):
        """
        This function plots the training-validation results of the Univarite model.
        
        Returns
        ----------
        A Plotly figure.
        """
        temp = pd.DataFrame({'Training':self.history.history['loss'],
                             'Validation':self.history.history['val_loss']})
        
        fig = px.line(temp[['Training','Validation']])
        fig['layout'] = dict(title='Training-Validation Results', titlefont=dict(size=25),
                             xaxis=dict(title='Epochs', titlefont=dict(size=16)),
                             yaxis=dict(title='Losses', titlefont=dict(size=16)))
        fig.show()
        
    def plot_results(self):
        """
        This function plots the prediction of the trained model of the training-validation data.
        
        Returns
        ----------
        A Plotly figure.
        """

        if self.is_scale:
            actual = self.scaler_label.inverse_transform(self.y)
            pred = self.model.predict(self.X)
            pred = self.scaler_label.inverse_transform(pred)

        else:
            actual = self.y
            pred = self.model.predict(self.X)
        
        temp = pd.DataFrame({'Actual':actual.flatten(), 'Forecasting':pred.flatten()})
        
        temp.index = self.index[self.w_size-1:][-len(temp):] # w_size-1 is lost data
        
        fig = go.Figure()
        
        fig['layout'] = dict(title='Forcasting Train-Test data',titlefont=dict(size=20),
                             xaxis=dict(title='Date', titlefont=dict(size=18)),
                             yaxis=dict(title='Value', titlefont=dict(size=18),))
        
        fig.add_scatter(x=temp.index, y=temp['Actual'], name='Actual')
        fig.add_vline(x=temp.index[self.split_point], line_width=3, line_dash='dash', line_color='black')
        fig.add_scatter(x=temp.index, y=temp['Forecasting'], name='Forecasting')
        
        fig.show()
        
    def forecast(self, test):
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
        self.test = test
        self.sequence = test.values
        self.test_index = test.index
        #take the date to start forecasting from
        while True:
            start_forecasting = str(input('Enter the date to start forecasting from ({} to {})'.\
                                          format(self.test_index.min().strftime('%Y-%m-%d'), 
                                                 self.test_index.max().strftime('%Y-%m-%d'))))

            if start_forecasting not in self.test_index:
                print('Date is not in range of data')
            else:
                break

        start = self.test_index.get_loc(start_forecasting)
        self.forecasting_data = self.test.iloc[-200-start:] 
        
        self.test_X, self.test_y = self.split_sequence(self.forecasting_data)  
        
        if self.is_scale:
            actual_data = self.scaler_label.inverse_transform(self.test_y)
            pred_data = self.model.predict(self.test_X)
            pred_data = self.scaler_label.inverse_transform(pred_data)
        else:
            actual_data = y
            pred_data = self.model.predict(self.test_X)

        temp = pd.DataFrame({'Actual':actual_data.flatten() ,'Forecasting':pred_data.flatten()})

        fig = go.Figure()

        fig['layout'] = dict(title='Forcasting From {} to {}'.format(start_forecasting,
                                                                     self.test_index[-1].strftime('%Y-%m-%d')),
                             titlefont=dict(size=20),
                             xaxis=dict(title='Date', titlefont=dict(size=18)),
                             yaxis=dict(title='Value', titlefont=dict(size=18),))

        fig.add_scatter(x=self.test_index[-200+start:], y=temp.iloc[-200+start:]['Actual'], name='Actual')
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
    Methode for modeling('single dense' or 'multi dense')
    Single dense is a MLP model with only one MLP.
    Multi dense is a MLP model with one dense layer for each submodel.
    """
    def __init__(self, training, methode):
        
        self.methode = re.sub(r'[^\w]', ' ', methode).lower()
        
        if self.methode not in ['single dense', 'multi dense']:
            raise ValueError('Invalid {}. Expecting single dense or multi dense.'.format(self.methode))
        
        self.data = training
        self.sequence = training.values
        self.index = training.index
        self.num_cols = len(training.columns)
        
        print('Your Multivarite object is ceated!')

    # split a univariate sequence into samples
    @tf.autograph.experimental.do_not_convert
    def split_sequences(self, data, train_test=True):
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
        if self.methode=='single dense':
            dataset = dataset.map(lambda window: (window[:,:-1], window[-1:,-1:][0]))
            self.X = np.array([x.numpy() for x, _ in dataset])
            self.y = np.array([y.numpy() for _, y in dataset])
            self.n_input = self.X.shape[1] * self.X.shape[2]
            self.X = self.X.reshape((self.X.shape[0], self.n_input))
            #scaling the data 
            if self.is_scale:
                self.X, self.y = self.scale_data(self.X, self.y)
        
        else:
            dataset = dataset.map(lambda window: (window[:-1], window[-2:][0][-1]))#if concate(feat_n,out)==>take right_button
            self.X = np.array([x.numpy() for x, _ in dataset])
            self.y = np.array([y.numpy() for _, y in dataset])
            #scaling the data 
            if self.is_scale:
                self.X, self.y = self.scale_data(self.X, self.y)
            
        if train_test:
            # Split data using train proportion of (70% - 30%)
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
  
    def build_dense_layers(self, units, w_size, epochs=100, lr=0.001, batch=32, scale=False):
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
        self.batch = batch
        self.epochs = epochs
        self.w_size = w_size
        self.is_scale = scale
        self.units = units
        callback = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,
                                                 patience=5, verbose=0, mode='auto')

        if self.methode=='single dense':
            
            # split the sequence into samples
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_sequences(self.sequence)
                
            # Build the model
            self.model = Sequential()
            self.model.add(Dense(units=self.units, kernel_initializer='normal', input_dim=self.n_input, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=100, kernel_initializer='normal', activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=8, kernel_initializer='normal', activation='relu')) 
            self.model.add(Dense(1))
            
            # Compile model
            opt = Adam(learning_rate=self.lr)
            self.model.compile(loss='mean_squared_error', optimizer=opt)
            self.history = self.model.fit(x=self.X_train, y=self.y_train, 
                                          validation_data=(self.X_test, self.y_test),
                                          callbacks=callback,
                                          batch_size=self.batch,
                                          epochs=self.epochs)        
        else:
            # Data Prepration
            # convert into input/output
            self.X, self.y = self.split_sequences(self.sequence, train_test=False)
            self.split_point = int(0.7 * len(self.X)) 
            # Split the data (each feature is feeded to one submodel) and split each to train-test
            self.data = {}
            for i in range(self.num_cols-1):
                self.data['X_%d'%i] = {}
                self.data['X_%d'%i]['train'] = self.X[:, :, i][:self.split_point]
                self.data['X_%d'%i]['test'] =  self.X[:, :, i][self.split_point:]

            self.y_train, self.y_test = self.y[:self.split_point], self.y[self.split_point:]
        
            # build the submodels(model for each feature)
            self.model_layers = {'inputs':{},'dense':{}}
            for i in range(self.num_cols-1):
                # input layer
                input_layer = Input(shape=(self.w_size-1,))
                self.model_layers['inputs']['input_%s'%i] = input_layer
                # dense layer
                dense_layer = Dense(self.units, activation='relu')
                self.model_layers['dense']['dense_%s'%i] = dense_layer(self.model_layers['inputs']['input_%s'%i])
                
            # merge input models
            merge = concatenate([dense for dense in self.model_layers['dense'].values()])
            output = Dense(64, activation='relu')(merge)
            output = Dense(1)(output)
            # connect input and output models
            self.model = Model(inputs=[inp for inp in self.model_layers['inputs'].values()], outputs=output)
            # Compile the model
            opt = Adam(learning_rate=self.lr)
            self.model.compile(optimizer='adam', loss='mse')
            # fit the model
            self.history = self.model.fit([self.data['X_%d'%col]['train'] for col in range(self.num_cols-1)], 
                                          self.y_train, epochs=self.epochs, verbose=1, 
                                          batch_size=self.batch,
                                          validation_data=([[self.data['X_%d'%col]['test'] for col in range(self.num_cols-1)]],
                                                           [self.y_test]),
                                          callbacks=callback,)
        
    def show_train_test_results(self):
        """
        This function plots the training-validation results of the Univarite model.
        
        Returns
        ----------
        A Plotly figure.
        """
        temp = pd.DataFrame({'Training':self.history.history['loss'],
                             'Validation':self.history.history['val_loss']})
        
        fig = px.line(temp[['Training','Validation']])
        fig['layout'] = dict(title='Training-Validation Results', titlefont=dict(size=25),
                             xaxis=dict(title='Epochs', titlefont=dict(size=16)),
                             yaxis=dict(title='Losses', titlefont=dict(size=16)))
        fig.show()
        
    def plot_results(self):
        
        if self.methode=='single dense':
            X, y = self.split_sequences(self.data, train_test=False)
            if self.is_scale:
                actual_data = self.scaler_label.inverse_transform(y)
                pred_data = self.model.predict(X)
                pred_data = self.scaler_label.inverse_transform(pred_data)
            else:
                actual_data = y
                pred_data = self.model.predict(self.X)
                
        else:
                        
            X = [self.X[:, :, i] for i in range(self.num_cols-1)]
            
            if self.is_scale:
                actual_data = self.scaler_label.inverse_transform(self.y)
                pred_data = self.model.predict(X)
                pred_data = self.scaler_label.inverse_transform(pred_data)
            else:
                actual_data = self.y
                pred_data = model.predict(self.X)
                
                
        temp = pd.DataFrame({'Actual':actual_data.reshape(-1, ), 'Forecasting':pred_data.reshape(-1)})
        temp.index = self.index[self.w_size-1:]
        
        fig = go.Figure()
        fig['layout'] = dict(title='Forcasting Train-Test data',titlefont=dict(size=20),
                             xaxis=dict(title='Date', titlefont=dict(size=18)),
                             yaxis=dict(title='Value', titlefont=dict(size=18),))
        
        fig.add_scatter(x=temp.index, y=temp['Actual'], name='Actual')
        fig.add_vline(x=temp.index[self.split_point], line_width=3, line_dash='dash', line_color='black')
        fig.add_scatter(x=temp.index, y=temp['Forecasting'], name='Forecasting')
        
        fig.show()
        
        
    def forecast(self, test):
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
        self.test = test
        self.sequence = test.values
        self.test_index = test.index
        #take the date to start forecasting from
        while True:
            start_forecasting = str(input('Enter the date to start forecasting from ({} to {})'.\
                                          format(self.test_index.min().strftime('%Y-%m-%d'), 
                                                 self.test_index.max().strftime('%Y-%m-%d'))))

            if start_forecasting not in self.test_index:
                print('Date is not in range of data')
            else:
                break

        start = self.test_index.get_loc(start_forecasting)
        self.forecasting_data = self.test.iloc[-200-start:] 
        
        self.test_X, self.test_y = self.split_sequences(self.forecasting_data, train_test=False)  
        
        if self.methode=='multi dense':
            self.test_X = [self.test_X[:, :, i] for i in range(self.num_cols-1)]

        if self.is_scale:
            actual_data = self.scaler_label.inverse_transform(self.test_y)
            pred_data = self.model.predict(self.test_X)
            pred_data = self.scaler_label.inverse_transform(pred_data)
        else:
            actual_data = y
            pred_data = self.model.predict(self.test_X)

        temp = pd.DataFrame({'Actual':actual_data.flatten() ,'Forecasting':pred_data.flatten()})

        fig = go.Figure()

        fig['layout'] = dict(title='Forcasting From {} to {}'.format(start_forecasting,
                                                                     self.test_index[-1].strftime('%Y-%m-%d')),
                             titlefont=dict(size=20),
                             xaxis=dict(title='Date', titlefont=dict(size=18)),
                             yaxis=dict(title='Value', titlefont=dict(size=18),))

        fig.add_scatter(x=self.test_index[-200+start:], y=temp.iloc[-200+start:]['Actual'], name='Actual')
        fig.add_vline(x=start_forecasting, line_width=3, line_dash='dash', line_color='black')
        fig.add_scatter(x=self.test_index[start:], y=temp.iloc[start:]['Forecasting'], name='Forecasting')
        fig.show()
        
class Create_Multivariate_Multple_Parallel:
    """
    This class creates and train a MLP model for multivariate TS.
    ...
    Attributes
    ----------
    data : Pandas dataframe
    Data to fit the model with.
    methode :  str
    Methode for modeling('vector output' or 'multi output')
    Single dense is a MLP model with only one MLP.
    Multi dense is a MLP model with one dense layer for each submodel.
    """
    def __init__(self, training, methode):
        
        self.methode = re.sub(r'[^\w]', ' ', methode).lower()
        
        if self.methode not in ['vector output', 'multi output']:
            raise ValueError('Invalid {}. Expecting vector output or multi output.'.format(self.methode))
        
        self.sequence = training.values
        self.index = training.index
        self.columns = [col[0].capitalize() for col in training.columns]
        self.num_cols = len(training.columns)
        
        print('Your Multivarite object is ceated!')
        
    # split a multivariate sequence into samples
    @tf.autograph.experimental.do_not_convert
    def split_sequences(self, sequence, train_test=True):
        
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
        
        # flatten input
        self.n_input = self.X.shape[1] * self.X.shape[2]
        self.X = self.X.reshape((self.X.shape[0], self.n_input))
        
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
        
    def build_dense_layers(self, units, w_size, epochs=100, lr=0.001, batch=32, scale=False):
        
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.w_size = w_size
        self.is_scale = scale
        self.units = units
        
        # split the sequence into samples
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_sequences(self.sequence, train_test=True)
        
        callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0,
                                                 patience=5,
                                                 verbose=0, 
                                                 mode='auto')
        # determine the number of outputs
        self.n_output = self.y_train.shape[1]                   
#         return self.X_train, self.y_train

        if self.methode=='vector output':    
            # Build the model
            self.model = Sequential()
            self.model.add(Dense(units=self.units, input_dim=self.n_input, kernel_initializer='normal', activation='relu'))
#             self.model.add(Dropout(0.2))
#             self.model.add(Dense(units=100, kernel_initializer='normal', activation='relu'))
#             self.model.add(Dropout(0.2))
#             self.model.add(Dense(units=100, kernel_initializer='normal', activation='relu')) 
            self.model.add(Dense(self.n_output))
            
            # Compile model
            opt = Adam(learning_rate=self.lr)
            self.model.compile(loss='mean_squared_error', optimizer=opt)

            self.history = self.model.fit(x=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test),
                                          batch_size=self.batch,  
                                          callbacks=callback,
                                          epochs=self.epochs)
        
        else:
            
            # define model
            self.visible = Input(shape=(self.n_input,))
            self.dense = Dense(100, activation='relu')(self.visible)
            
            # split the output(for each output layer)
            self.outputs = {'output_%d'%i:{} for i in range(self.n_output)}
            
            # create output layer each for one TS
            self.output_layers = {'output_%d'%i:None for i in range(self.n_output)}
            
            for i in range(self.n_output):
                self.outputs['output_%d'%i]['train'] = self.y[:, i].reshape((self.y.shape[0], 1))[:self.split_point]
                self.outputs['output_%d'%i]['test'] = self.y[:, i].reshape((self.y.shape[0], 1))[self.split_point:]
                self.output_layers['output_%d'%i] = Dense(1)(self.dense)
            
            # tie together
            self.model = Model(inputs=self.visible, outputs=[out for out in self.output_layers.values()])

            # Compile model
            opt = Adam(learning_rate=self.lr)
            self.model.compile(loss='mean_squared_error', optimizer=opt)

            self.history = self.model.fit(x=self.X_train, 
                                          y=[self.outputs['output_%d'%i]['train'] for i in range(self.n_output)],
                                          validation_data=(self.X_test, 
                                                           [self.outputs['output_%d'%i]['test'] for i in range(self.n_output)]),
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
            X, y = self.split_sequences(self.sequence, train_test=False)
            
            if self.is_scale:
                actual = self.scaler_label.inverse_transform(y)
                pred = self.model.predict(X)
                pred = self.scaler_label.inverse_transform(pred)
                
            else:
                actual = y
                pred = self.model.predict(self.X)
    
            for i in range(self.num_cols):
                temp = pd.DataFrame({'Actual':actual[:,i].flatten(), 'Forecasting':pred[:,i].flatten()})
                fig=px.line(temp)
                fig['layout'] = dict(title='Forcasting Train-Test data for {}'.format(self.columns[i]),
                                     titlefont=dict(size=20),
                                     xaxis=dict(title='Date', titlefont=dict(size=18)),
                                     yaxis=dict(title='Value', titlefont=dict(size=18)))
                fig.add_vline(x=temp.index[self.split_point], line_width=3, line_dash='dash', line_color='black')
                fig.show()
                
        else:
            pred_data = self.model.predict(self.X)
            actual_data = self.y
#             return pred_data, actual_data
            
            for i in range(self.n_output):
                temp = pd.DataFrame({'Actual':actual_data[:,i].flatten(),
                                     'Predictions':pred_data[i].flatten()})
                temp.index = self.index[:len(temp)]
                fig = px.line(temp)
                fig['layout']=dict(title=self.columns[i], titlefont=dict(size=18))
                fig.add_vline(x=temp.index[self.split_point], line_width=3, line_dash='dash', line_color='black')
                fig.show()


    def forecast(self, data):
        self.sequence = data.values
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

        start_forecasting = data.index.get_loc(start_forecasting)
        n = len(self.sequence[start_forecasting:])

        if self.methode=='vector output':    

            self.test_X, self.test_y = self.split_sequences(self.sequence, train_test=False)
            self.test_X = self.test_X.reshape((self.test_X.shape[0], self.n_input))
        
            if self.is_scale:
                actual = self.scaler_label.inverse_transform(self.test_y)
                pred = self.model.predict(self.test_X)
                pred = self.scaler_label.inverse_transform(pred)
            else:
                actual = y
                pred = self.model.predict(self.test_X)
            
            for i in range(self.num_cols):
                temp = pd.DataFrame({'Actual': actual[:,i].flatten(), 'Forecasting': pred[:,i].flatten()})
        
                fig = go.Figure()
                fig['layout'] = dict(title='Forcasting {} From {} to {}'.\
                                     format(self.columns[i],
                                            self.test_index[-n].strftime('%Y-%m-%d'),
                                            self.test_index[-1].strftime('%Y-%m-%d')),
                                     titlefont=dict(size=20),
                                     xaxis=dict(title='Date', titlefont=dict(size=18)),
                                     yaxis=dict(title='Value', titlefont=dict(size=18),))

                fig.add_scatter(x=self.test_index[-200-n:], y=temp.iloc[-200-n:]['Actual'], name='Actual')

                fig.add_vline(x=self.test_index[-n], line_width=3, line_dash='dash', line_color='black')

                fig.add_scatter(x=self.test_index[-n:], y=temp.iloc[-n:]['Forecasting'], name='Forecasting')

                fig.show()
                               
        else:
            
            # split the sequence into samples
            self.X, self.y = self.split_sequences(self.sequence, train_test=False)
            self.X = self.X.reshape((self.X.shape[0], self.n_input))

            actual_data = [self.y[:, i].reshape((self.y.shape[0], 1)) for i in range(self.n_output)]
            
            pred_data = self.model.predict(self.X)
            
            for i in range(self.n_output):
                
                temp = pd.DataFrame({'Actual':actual_data[i].flatten() ,
                                     'Forecasting':pred_data[i].flatten()})

                fig = go.Figure()

                fig['layout'] = dict(title='Forcasting {} From {} to {}'.\
                                     format(self.columns[i],
                                            self.test_index[-n].strftime('%Y-%m-%d'),
                                            self.test_index[-1].strftime('%Y-%m-%d')),
                                     titlefont=dict(size=20),
                                     xaxis=dict(title='Date', titlefont=dict(size=18)),
                                     yaxis=dict(title='Value', titlefont=dict(size=18),))

                fig.add_scatter(x=self.test_index[-200-n:], y=temp.iloc[-200-n:]['Actual'], name='Actual')

                fig.add_vline(x=self.test_index[-n], line_width=3, line_dash='dash', line_color='black')

                fig.add_scatter(x=self.test_index[-n:], y=temp.iloc[-n:]['Forecasting'], name='Forecasting')

                fig.show()
                
                
class Create_Univariate_Multi_Step_MLP:
    """
    This class creates and train a MLP model for Univariate Multi-Step Forecasting.
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
                          
    def build_dense_layers(self, n_steps_in, n_steps_out, units, scale=False, epochs=100, lr=0.03, batch=256):
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
        self.model.add(Dense(units=100, kernel_initializer='normal', activation='relu'))
#             self.model.add(Dropout(0.2))
#             self.model.add(Dense(units=100, kernel_initializer='normal', activation='relu')) 
        self.model.add(Dense(self.n_steps_out))
        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer=opt)

        self.history = self.model.fit(x=self.X_train, y=self.y_train, 
                                      validation_data=(self.X_test, self.y_test),
                                      batch_size=self.batch, 
                                      callbacks=callback,
                                      epochs=self.epochs, 
                                      verbose=1)

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
        while True:
            start_forecasting = str(input('Enter the date to start forecasting from ({} to {})'.\
                                          format(self.data.index.min().strftime('%Y-%m-%d'), 
                                                 self.data.index.max().strftime('%Y-%m-%d'))))

            if start_forecasting not in self.data.index:
                print('Date not in range of data')
            else:
                break
        
        start_forecasting_index = self.data.index.get_loc(start_forecasting)
        self.forecast_data = self.data[start_forecasting_index-self.n_steps_in:start_forecasting_index+self.n_steps_out]

        self.test_X, self.test_y = self.forecast_data[:self.n_steps_in], self.forecast_data[self.n_steps_in:]
        
        pred = self.model.predict([self.test_X.values.reshape((1, self.n_steps_in))]).flatten()
        pred = pd.Series(pred, index=self.forecast_data.index[self.n_steps_in:])

        start = abs(self.data.index.get_loc(start_forecasting) - self.n_steps_in)
        start = self.data.reset_index().loc[start]['Date'].strftime('%Y-%m-%d')
        
        end = self.data.index.get_loc(start_forecasting) + self.n_steps_out
        end = self.data.reset_index().loc[end]['Date'].strftime('%Y-%m-%d')
        
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
    def split_sequences(self, sequence, train_test=False, scale=False):
        
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
            # flatten input
            self.X = self.X.reshape((self.X.shape[0], self.n_input))
            # determine the number of outputs
            self.n_output = self.y.shape[1]  
            
        else:
            dataset = dataset.map(lambda window: (window[:self.n_steps_in,:], window[self.n_steps_in-1:self.w_size-1,:]))
            self.X = np.array([x.numpy() for x, _ in dataset])
            self.y = np.array([y.numpy() for _, y in dataset])
            # determine the number of outputs
            self.n_input = self.X.shape[1] * self.X.shape[2]   
            # flatten input
            self.X = self.X.reshape((self.X.shape[0], self.n_input))
            # determine the number of outputs
            self.n_output = self.y.shape[1] * self.y.shape[2]           
            # flatten input
            self.y = self.y.reshape((self.y.shape[0], self.n_output))
        
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
                          
    def build_dense_layers(self, n_steps_in, n_steps_out, units, scale=True, epochs=100, lr=0.003, batch=32):
        
        self.lr = lr
        self.batch = batch
        self.units = units
        self.epochs = epochs
        self.is_scale = scale
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.w_size = self.n_steps_in + self.n_steps_out
        # split the sequence into samples
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_sequences(self.sequence, train_test=True,
                                                                                    scale=self.is_scale)
        
        callback = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,
                                                 patience=5, verbose=0, mode='auto')
        
        if self.methode=='multiple input':    

            # Build the model        
            opt = Adam(learning_rate=self.lr)
            self.model = Sequential()
            self.model.add(Dense(self.units, kernel_initializer='normal', activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(8, kernel_initializer='normal', activation='relu'))
            self.model.add(Dense(self.n_output))
            # Compile model
            self.model.compile(loss='mean_squared_error', optimizer=opt)

            self.history = self.model.fit(x=self.X_train, 
                                          y=self.y_train, 
                                          validation_data=(self.X_test, self.y_test),
                                          callbacks=callback, 
                                          epochs=self.epochs,
                                          batch_size=self.batch)
                                
        else:
            # Build the model        
            opt = Adam(learning_rate=self.lr)
            self.model = Sequential()
            self.model.add(Dense(self.units, input_dim=self.n_input, kernel_initializer='normal', activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(64, kernel_initializer='normal', activation='relu'))
            self.model.add(Dense(self.n_output))

            # Compile model
            self.model.compile(loss='mean_squared_error', optimizer=opt)

            self.history = self.model.fit(x=self.X_train, y=self.y_train, 
                                          validation_data=(self.X_test, self.y_test),
                                          callbacks=callback,
                                          batch_size=self.batch, 
                                          epochs=self.epochs)
            
    def show_train_test_results(self):
        
        if self.methode=='multiple input':
            temp = pd.DataFrame({'Training':self.history.history['loss'], 'Validation':self.history.history['val_loss']})

            fig = px.line(temp)
            fig.show()
        
        
        else:
            columns = self.columns
            num_cols = self.num_cols
            temp = pd.DataFrame(self.history.history)

            fig = px.line(temp)
            fig.show()
    
    def forecast(self, data):
        
        self.sequence = data.values
        self.test_index = data.index    
        self.data = data
        
        while True:
            start_forecasting = str(input('Enter the date to start forecasting from ({} to {})'.\
                                          format(self.data.index.min().strftime('%Y-%m-%d'), 
                                                 self.data.index.max().strftime('%Y-%m-%d'))))

            if start_forecasting not in self.data.index:
                print('Date not in range of data')
            else:
                break
                
        start_forecasting_index = self.data.index.get_loc(start_forecasting)
        self.forecast_data = self.data[start_forecasting_index-self.n_steps_in : start_forecasting_index+self.n_steps_out]
        
        self.test_X, self.test_y = self.split_sequences(self.sequence)
        
        start = abs(self.data.index.get_loc(start_forecasting) - self.n_steps_in)
        start = self.data.reset_index().loc[start]['Date'][0].strftime('%Y-%m-%d')

        end = self.data.index.get_loc(start_forecasting) + self.n_steps_out
        end = self.data.reset_index().loc[end]['Date'][0].strftime('%Y-%m-%d')
        
        if self.methode=='multiple input':    
            if self.is_scale:
                pred = self.model.predict(self.test_X)
                pred = self.scaler_label.inverse_transform(pred)
                actual = self.scaler_label.inverse_transform(self.test_y)
            else:
                pred = self.model.predict(self.test_X)
                actual = self.test_y
            
            temp = pd.DataFrame({'Actual':actual.flatten(), 'Forecasting':pred.flatten()})            
            temp = temp.iloc[start_forecasting_index-self.n_steps_in : start_forecasting_index+self.n_steps_out]
            temp.index = self.forecast_data.index
            fig = go.Figure()
            fig['layout'] = dict(title='Forcasting From {} to {}'.format(start, end),
                                 titlefont=dict(size=20),
                                 xaxis=dict(title='Date', titlefont=dict(size=18)),
                                 yaxis=dict(title='Value', titlefont=dict(size=18),))
            fig.add_scatter(x=temp.index, y=temp['Actual'], name='Actual')
            fig.add_vline(x=start_forecasting, line_width=3, line_dash='dash', line_color='black')
            fig.add_scatter(x=temp.index[self.n_steps_in:], y=temp['Forecasting'][self.n_steps_in:], name='Forecasting')
            fig.show()
            
        else:
            if self.is_scale:
                pred = self.model.predict(self.test_X)
                pred = self.scaler_label.inverse_transform(pred)
                actual = self.scaler_label.inverse_transform(self.test_y)
            else:
                pred = self.model.predict(self.test_X)
                actual = self.test_y
            
            for i, col in enumerate(self.columns):

                temp = pd.DataFrame({'Actual':actual[:,i].flatten(), 'Forecasting':pred[:,i].flatten()})            
                temp = temp.iloc[start_forecasting_index-self.n_steps_in : start_forecasting_index+self.n_steps_out]
                temp.index = self.forecast_data.index

                fig = go.Figure()
                fig['layout'] = dict(title='Forcasting From {} to {}'.format(start, end),
                                     titlefont=dict(size=20),
                                     xaxis=dict(title='Date', titlefont=dict(size=18)),
                                     yaxis=dict(title='Value', titlefont=dict(size=18),))
                fig.add_scatter(x=temp.index, y=temp['Actual'], name='Actual')
                fig.add_vline(x=start_forecasting, line_width=3, line_dash='dash', line_color='black')
                fig.add_scatter(x=temp.index[self.n_steps_in:], y=temp['Forecasting'][self.n_steps_in:], name='Forecasting')
                fig.show()