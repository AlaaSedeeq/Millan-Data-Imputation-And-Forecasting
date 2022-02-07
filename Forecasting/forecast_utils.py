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
from statsmodels.tsa.seasonal import seasonal_decompose, STL
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

# Functions 
def ts_decomposition(df:pd.DataFrame, column:str, title:str, period=144, model='additive', return_results=False):
    
    decomposition = seasonal_decompose(df[column], period=period, model=model)
    seasonal, trend, resid = decomposition.seasonal, decomposition.trend, decomposition.resid
    
    decomposition = pd.DataFrame(index=df.index)
    decomposition['Sequence'] = df[column]
    decomposition['Seasonal'] = seasonal.values
    decomposition['Trend'] = trend.values
    decomposition['Residuals'] = resid.values

    fig = make_subplots(rows=4, cols=1, subplot_titles=['Sequence','Seasonal','Trend','Residuals'])
    fig.append_trace((go.Scatter(x=decomposition.index, y=decomposition['Sequence'], showlegend=False)),1,1)
    fig.append_trace((go.Scatter(x=decomposition.index, y=decomposition['Seasonal'], showlegend=False)),2,1)
    fig.append_trace((go.Scatter(x=decomposition.index, y=decomposition['Trend'], showlegend=False)),3,1)
    fig.append_trace((go.Scatter(x=decomposition.index, y=decomposition['Residuals'], showlegend=False)),4,1)
    
    if return_results:
        return seasonal, trend, resid
    
    fig.show()
    
def plot_line(data, y, title='', range_x=None):
    fig = px.line(data, y=y, range_x=range_x)

    fig['layout'] = dict(title=dict(text=title,  font=dict(size=20), xanchor='auto'),
                         xaxis=dict(title='Date', titlefont=dict(size=18), rangeslider_visible=True),
                         yaxis=dict(title='Value', titlefont=dict(size=18)))
    fig.show()
    
    
def plot_hist(data, title='', range_x=None):
    fig = px.histogram(data, range_x=range_x)
    fig['layout'] = dict(title=dict(text=title.title(), font=dict(size=20), xanchor='auto'),
                         xaxis=dict(title='Date', titlefont=dict(size=18)),
                         yaxis=dict(title='Count', titlefont=dict(size=18)))
    fig.show()

    
def test_stationarity(df, series, window_size, title='', ret_values=None):
    
    # Determing rolling statistics
    rolmean = df[series].rolling(window = window_size, center = False).mean().dropna() #Checkif our data has constant mean
    rolstd = df[series].rolling(window = window_size, center = False).std().dropna()   #Checkif our data has constant variance
    
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
    print('Failed to Reject Ho ---> Time Series is Non-Stationary' if dfoutput[1] > 0.05\
          else "Reject Ho ---> Time Series is Stationary")
           
    # Plot rolling statistics:
    fig = plt.figure(figsize=(18,6))
    plt.plot(df[series].values)
    plt.plot(rolmean.values)
    plt.plot(rolstd.values)
    plt.legend(['TS', 'Rolling Mean', 'Rolling std'])
    plt.show()
    
#     fig = go.Figure()
#     fig.add_scatter(x=tuple(range(len(df[series]))), y=df[series].values, name='TS Data')
#     fig.add_scatter(x=tuple(range(len(rolmean.values))), y=rolmean.values, name='Rolling Mean')
#     fig.add_scatter(x=tuple(range(len(rolstd.values))), y=rolstd.values, name='Rolling std')
#     fig['layout'] = dict(title=title.title(), titlefont=dict(size=20),
#                          xaxis=dict(title='Range', titlefont=dict(size=18)),
#                          yaxis=dict(title='Values', titlefont=dict(size=18)))
#     fig.show()
    
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
    

# To convert time column into a datetime
def convert(t):
    index = dt.datetime.fromtimestamp(t / 1e3) - dt.timedelta(hours = 1)
    index = pd.to_datetime(index)
    return index


def neighbor_cells(data:np.ndarray, neighbor_radius:int, cell_center:list, num_samples=None, uni_variant=True):
    if not num_samples:
        num_samples = data.shape[0]
    # the cell center coordination.
    cell_center = np.array(cell_center)
    # check if cell center coordination is only one value.
    if len(cell_center) == 1:
            # ensure it is in the range of data's boundaries.
            # if not in data's boundaries, reject it.
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
        # if not data's boundaries, reject it.
        if (cell_center[0]>data.shape[1]-1) or (cell_center[1]>data.shape[2]-1):
            raise(Exception('Cell target must be a tuple has the position of the target cell within the grid'))
        # if each value is in the range of data's boundaries, take them as x and y.
        else:
            row_number, column_number = cell_center[0], cell_center[1]
#     return num_samples
    # the output data frame
    multi_grids_data = pd.DataFrame()
    multi_grids_data['time'] = pd.RangeIndex(start=1383260400000, stop=1383260400000+num_samples*10000*60, step=10000*60)
    multi_grids_data["time"] = multi_grids_data["time"].apply(lambda t: dt.datetime.fromtimestamp(t / 1e3)-\
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
#         return num_samples
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

def divide_data(iter_num, df, day_1):
    """
    function to divide the data set into training and prediction sets
    """

    day = 144
    month = 30*144
    start_t = iter_num*day
    end_t_day_1 = start_t + month
    
    # For training set
    X_train = df[start_t : end_t_day_1]
    
    data = {'Training':X_train, 
            'Forecasting':{'Day_1':None, 'Day_2':None, 'Day_3':None, 'Week':None}}
    
    # For prediction set
    if iter_num <= (day_1-6):
        # 1 Day
        data['Forecasting']['Day_1'] = df[end_t_day_1 : end_t_day_1 + day]
        # 2 Day
        data['Forecasting']['Day_2'] = df[end_t_day_1 : end_t_day_1 + 2*day]
        # 3 Day
        data['Forecasting']['Day_3'] = df[end_t_day_1 : end_t_day_1 + 3*day]
        # 7 Day
        data['Forecasting']['Week'] = df[end_t_day_1 : end_t_day_1 + 7*day]
        
    elif iter_num <= (day_1-3):
        # 1 Day
        data['Forecasting']['Day_1'] = df[end_t_day_1 : end_t_day_1 + day]
        # 2 Day
        data['Forecasting']['Day_2'] = df[end_t_day_1 : end_t_day_1 + 2*day]
        # 3 Day
        data['Forecasting']['Day_3'] = df[end_t_day_1 : end_t_day_1 + 3*day]
        
    elif iter_num <= (day_1-2):
        # 1 Day
        data['Forecasting']['Day_1'] = df[end_t_day_1 : end_t_day_1 + day]
        # 2 Day
        data['Forecasting']['Day_2'] = df[end_t_day_1 : end_t_day_1 + 2*day]
        
    elif iter_num <= (day_1):
        # 1 Day
        data['Forecasting']['Day_1'] = df[end_t_day_1 : end_t_day_1 + day]
        
    return data

def EXP_Smoothing(X_train, forecast_data, args):
    
    trend = args['trend']
    seasonal = args['seasonal']
    seasonal_periods = args['seasonal_periods']
    errors = {'nrmse':None, 'mape':None}
    forecasting_steps = len(forecast_data)

    with catch_warnings():
        filterwarnings("ignore")
        EXP_smooth = ExponentialSmoothing(X_train, trend='mul',seasonal='mul',seasonal_periods=144).fit()
        EXP_smooth_pred = EXP_smooth.forecast(forecasting_steps)
    
    forecast_data['Predictions'] = EXP_smooth_pred.values
    
    errors['nrmse'] = nrmse(forecast_data)
    errors['mape'] = mape(forecast_data)

    return forecast_data, errors

def VAR_model(data, Results, day_week, model_name, itr):

    forecasting_steps = len(data['Forecasting'][day_week])

    if model_name.lower()=='var':
        
        model = VAR(data['Training'])
        model_fitted = model.fit()
        lag_order = model_fitted.k_ar
        results = model_fitted.forecast(data['Training'].values[-lag_order:], forecasting_steps)

    elif model_name.lower()=='stl-var':
        
        for grid in Results.keys():
            # decomposition
            decomposition = STL(data['Training'][grid], period=7*144).fit()
            seasonal, trend, resid = decomposition.seasonal, decomposition.trend, decomposition.resid
            # forecast only (trend + noise)
            data['Training']['%s_Seasonal'] = seasonal
            data['Training']['%s_Trend_Noise'] = trend + resid

        # Seasonal
        s_c = ['%s'%c for c in data['Training'].columns if 'Seasonal' in c]
        s_model = VAR(data['Training'][s_c])
        s_model_fitted = s_model.fit()
        s_lag_order = s_model_fitted.k_ar

        # Trend + Noise
        tn_c = ['%s'%c for c in data['Training'].columns if 'Seasonal' in c]
        tn_model = VAR(data['Training'][tn_c])
        tn_model_fitted = tn_model.fit()
        tn_lag_order = tn_model_fitted.k_ar

        results = model_fitted.forecast(data['Training'][s_c].values[-s_lag_order:], forecasting_steps)+\
                  model_fitted.forecast(data['Training'][tn_c].values[-tn_lag_order:], forecasting_steps)
        
    for i, grid in enumerate(Results.keys()):
        Results[grid][model_name][day_week][str(itr+1)]['results'] = results[:, i]
        errors = {'nrmse': nrmse(results[:, i]), 'mape' : mape(results[:, i])}
        Results[grid][model_name][day_week][str(itr+1)]['results'] = errors

    return Results

def create_acf_pacf(series, series_name=''):
    
    fig, axes = plt.subplots(1, 2, figsize=(20,5))
        
    sm.graphics.tsa.plot_acf(series, 
                             title='Autocorrelation for %s'%(series_name),
                             lags=40, ax=axes[0])
    
    sm.graphics.tsa.plot_pacf(series,
                              title='Partial Autocorrelation for %s'%(series_name),
                              lags=40, ax=axes[1])
    
    plt.show()
    
def forecasting(model, s_model, tn_model, iter_num, day_week, model_name, data, Results, steps, grid):

    if model:
        results = pd.DataFrame({'Predictions':model.predict(steps),
                                'Target':data['Forecasting']['%s'%(day_week)]['Target'].to_numpy().flatten()})

        errors = errors = {'nrmse': nrmse(results), 'mape' : mape(results)}
        Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name]['%s'%(day_week)][str(iter_num+1)]['results'] = results
        Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name]['%s'%(day_week)][str(iter_num+1)]['errors'] = errors

    elif (s_model and tn_model):
        results = pd.DataFrame({'Predictions':s_model.predict(steps)+tn_model.predict(steps) if 'arima' in model_name.lower()\
                                         else s_model.forecast(steps)+tn_model.forecast(steps),
                                'Target':data['Forecasting']['%s'%(day_week)]['Target']})
        errors = {'nrmse': nrmse(results), 'mape' : mape(results)}
        Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name]['%s'%(day_week)][str(iter_num+1)]['results'] = results
        Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name]['%s'%(day_week)][str(iter_num+1)]['errors'] = errors

    return Results

def ml_forecast(iter_num, data, model_name, model_args, day_week, Results, grid):
        
    models = {'svr':SVR, 
              'lasso':Lasso,
              'elasticnet': ElasticNet,
              'knn':KNeighborsRegressor,
              'linearregression':LinearRegression}
    
    regressor = models[model_name.lower().replace('stl-','').lower()](**model_args)
    
    if 'stl' in model_name.lower():
        forecaster = TransformedTargetForecaster([("deseasonalize", Deseasonalizer(model="multiplicative", sp=144)),
                                                  ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=3))),
                                                  ("forecast",make_reduction(regressor,
                                                                             scitype="tabular-regressor", 
                                                                             window_length=144, strategy="recursive"))])
    else:
        forecaster = make_reduction(regressor, window_length=144, strategy="recursive")
    
    train = data['Training']['Target']
    
    forecaster.fit(train)

    # Forecasting 1-Day
    forecast = data['Forecasting']['Day_1']['Target']
    fh = ForecastingHorizon(forecast.index, is_relative=False)
    y_pred = forecaster.predict(fh)
    results = pd.DataFrame({'Predictions':y_pred, 'Target':forecast})
    errors = {'nrmse': nrmse(results), 'mape' : mape(results)}
    Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name.lower()]['%s'%day_week][str(iter_num+1)]['results'] = results
    Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name.lower()]['%s'%day_week][str(iter_num+1)]['errors'] = errors

    # Forecasting 2-Day
    if iter_num < (day_1-1):
        forecast = data['Forecasting']['Day_2']['Target']
        fh = ForecastingHorizon(forecast.index, is_relative=False)
        fh = ForecastingHorizon(forecast.index, is_relative=False)
        y_pred = forecaster.predict(fh)
        results = pd.DataFrame({'Predictions':y_pred, 'Target':forecast})
        errors = {'nrmse': nrmse(results), 'mape' : mape(results)}
        Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name.lower()]['%s'%day_week][str(iter_num+1)]['results'] = results
        Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name.lower()]['%s'%day_week][str(iter_num+1)]['errors'] = errors
    
    # Forecasting 3-Day
    elif iter_num < (day_1-2):
        forecast = data['Forecasting']['Day_2']['Target']
        fh = ForecastingHorizon(forecast.index, is_relative=False)
        fh = ForecastingHorizon(forecast.index, is_relative=False)
        y_pred = forecaster.predict(fh)
        results = pd.DataFrame({'Predictions':y_pred, 'Target':forecast})
        errors = {'nrmse': nrmse(results), 'mape' : mape(results)}
        Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name.lower()]['%s'%day_week][str(iter_num+1)]['results'] = results
        Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name.lower()]['%s'%day_week][str(iter_num+1)]['errors'] = errors
    
    # Forecasting 1-Week
    elif iter_num < (day_1-6): 
        forecast = data['Forecasting']['Day_2']['Target']
        fh = ForecastingHorizon(forecast.index, is_relative=False)
        fh = ForecastingHorizon(forecast.index, is_relative=False)
        y_pred = forecaster.predict(fh)
        results = pd.DataFrame({'Predictions':y_pred, 'Target':forecast})
        errors = {'nrmse': nrmse(results), 'mape' : mape(results)}
        Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name.lower()]['%s'%day_week][str(iter_num+1)]['results'] = results
        Results['Grid(%d,%d)'%(grid[0], grid[1])]['%s'%model_name.lower()]['%s'%day_week][str(iter_num+1)]['errors'] = errors
    
    return Results

class Create_Univariate_Multi_Step_MLP:
    
    def __init__(self, training):
        
        self.sequence = training.values
        self.index = training.index
                
    # split a univariate sequence into samples
    def split_sequence(self, data, train_test=False, scale=False):

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

    def build(self, n_steps_in, n_steps_out, units, scale=False, epochs=100, lr=0.03, batch=256, verbose=0):
        
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
#         self.model.add(Dense(units=self.units/2, kernel_initializer='normal', activation='relu'))
#         self.model.add(Dropout(0.2))
#         self.model.add(Dense(units=self.units/2, kernel_initializer='normal', activation='relu')) 
        self.model.add(Dense(self.n_steps_out))
        
        # Compile model
        self.model.compile(loss='mse', optimizer=opt)

        self.history = self.model.fit(x=self.X_train, 
                                      y=self.y_train, 
                                      validation_data=(self.X_test, self.y_test),
                                      batch_size=self.batch, 
                                      callbacks=callback,
                                      epochs=self.epochs, 
                                      verbose=verbose)


    def forecast(self, data):
        X, y = self.split_sequence(data.values)
        pred = self.model.predict([X.reshape((1, self.n_steps_in))]).flatten()
        pred = self.scaler_features.inverse_transform([pred])
        pred = pd.Series(pred.flatten(), index=data.index[-self.n_steps_out:])
        return pred
    
    
class Create_Univariate_Multi_Step_CNN:
    def __init__(self, training):
        
        self.sequence = training.values
        self.index = training.index
    # split a univariate sequence into samples
    def split_sequence(self, data):
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
        
        if self.is_scale:
            self.X, self.y = self.scale_data(self.X, self.y)
            
        return self.X, self.y   
      
    def scale_data(self, X, y):

        self.scaler_features = MinMaxScaler().fit(X.reshape(-1, 1))
        self.scaled_features = self.scaler_features.transform(X.reshape(-1, 1)).reshape(np.shape(X))

        self.scaler_label = MinMaxScaler().fit(np.array(y.reshape(-1, 1)))
        self.scaled_label = self.scaler_label.transform(y.reshape(-1, 1)).reshape(np.shape(y))
        
        return self.scaled_features, self.scaled_label
        
    def build(self, n_steps_in, n_steps_out, filter_num=32, kernel_size=2, pool_size=2,
                  units=100, epochs=100, lr=0.03, batch = 256, scale=False, verbose=0):
        
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
     
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        self.n_features = 1
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], self.n_features))
        
        # Split data using train proportion of (70% - 30%)
        self.split_point = int(0.7 * len(self.X)) 
        self.X_train, self.X_test = self.X[:self.split_point], self.X[self.split_point:]
        self.y_train, self.y_test = self.y[:self.split_point], self.y[self.split_point:]
        
        
        # Build the model
        self.model = Sequential()
        self.model.add(Conv1D(filters=self.filter_num, kernel_size=self.kernel_size, activation='relu',
                              kernel_initializer='normal', input_shape=(self.n_steps_in, self.n_features)))
        self.model.add(MaxPooling1D(pool_size=self.pool_size))
        self.model.add(Flatten())
        self.model.add(Dense(self.units, activation='relu'))
        self.model.add(Dense(self.n_steps_out))
        
        # define the optimizer
        opt = Adam(learning_rate=self.lr)
        
        # Compile model
        self.model.compile(optimizer=opt, loss='mse')
        
        self.history = self.model.fit(x=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test),
                                      batch_size=self.batch, 
                                      callbacks=callback,
                                      epochs=self.epochs,
                                      verbose=verbose)

    def forecast(self, data):
        X, y = self.split_sequence(data.values)
        pred = self.model.predict([X.reshape((1, self.n_steps_in))]).flatten()
        pred = self.scaler_features.inverse_transform([pred])
        pred = pd.Series(pred.flatten(), index=data.index[-self.n_steps_out:])
        return pred
    
def plot_forecasting(real:pd.DataFrame, Results:dict, legend:list, models:list, w:int, days_weeks='Days', previous_steps=300):
    
    if len(models)==1:
        temp = Results['%s'%models[0]]['%s'%days_weeks]['%d'%w]['results'].rename(columns={'Predictions':'%s'%models[0]})\
               .reset_index()

        plt.figure(figsize=(20,8))
        plt.title('Forecasting %s %d'%(days_weeks[:-1], w), fontsize=20)
        plt.plot(real['Target'][-previous_steps:], lw=2.5, label='Actual', color='#56396c')
        plt.plot(temp[['%s'%models[0],'time']].set_index('time'), lw=2.5, label='%s'%legend[0])
        max_ = temp.drop(['Target','time'],axis=1).values.max()+50
        min_ = temp.drop(['Target','time'],axis=1).values.min()-50
        plt.vlines(x=temp['time'].min(),ymin=min_, ymax=max_, linestyle='dashed', color='#4a4e4d', label='Start Forecasting')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Value', fontsize=18)
        plt.legend()
        plt.show()
        
    else:
        temp = Results['%s'%models[0]]['%s'%days_weeks]['%d'%w]['results'].rename(columns={'Predictions':'%s'%models[0]})\
               .reset_index()
        
        plt.figure(figsize=(20,8))
        plt.title('Forecasting %s %d'%(days_weeks[:-1], w), fontsize=20)
        plt.plot(real['Target'][-previous_steps:], lw=2.5, color='#56396c', alpha=0.7, label='Actual')
        plt.plot(temp[['%s'%models[0],'time']].set_index('time'), label='%s'%legend[0])
        
        for i, model in enumerate(models[1:]):
            data = Results['%s'%model]['%s'%days_weeks]['%d'%w]['results'].rename(columns={'Predictions':'%s'%model})['%s'%model]
            plt.plot(data, label='%s'%legend[i+1])

        max_ = real['Target'][-previous_steps:].max()+50
        min_ = real['Target'][-previous_steps:].min()-50
        plt.vlines(x=temp['time'].min(),ymin=min_, ymax=max_, linestyle='dashed', color='#4a4e4d', label='Start Forecasting')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Value', fontsize=18)
        plt.legend()
        plt.show()
        
        
def dict_to_dataframe(Results:dict, forecast_grids:list, day_1:int=21):
    
    # Mean Mape & NRMSE for each Model for 1-Day, 2-Day, 3-Day 7-Day forecasting. 

    columns=[('1-Day', 'MAPE'), ('1-Day', 'NRMSE'), 
             ('2-Day', 'MAPE'), ('2-Day', 'NRMSE'), 
             ('3-Day', 'MAPE'), ('3-Day', 'NRMSE'),
             ('Week', 'MAPE'), ('Week', 'NRMSE')]

    grids_errors = {'Grid(%d,%d)'%(grid[0], grid[1]) : None for grid in forecast_grids}
    
    models = Results[list(Results.keys())[0]].keys()
    
    for grid in forecast_grids:

        grids_errors['Grid(%d,%d)'%(grid[0], grid[1])] = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns),
                                                                      index=map(lambda x:x.title(), models))

        grid_df = Results['Grid(%d,%d)'%(grid[0], grid[1])]

        for model in models:
            grids_errors['Grid(%d,%d)'%(grid[0], grid[1])].loc['%s'%model.title()] = \
            [np.mean([grid_df['%s'%model]['Day_1']['%s'%(str(i+1))]['errors']['mape'] for i in range(day_1)]),   # 1-Day
             np.mean([grid_df['%s'%model]['Day_1']['%s'%(str(i+1))]['errors']['nrmse'] for i in range(day_1)]),

             np.mean([grid_df['%s'%model]['Day_2']['%s'%(str(i+1))]['errors']['mape'] for i in range(day_1-1)]), # 2-Day
             np.mean([grid_df['%s'%model]['Day_2']['%s'%(str(i+1))]['errors']['nrmse'] for i in range(day_1-1)]),

             np.mean([grid_df['%s'%model]['Day_3']['%s'%(str(i+1))]['errors']['mape'] for i in range(day_1-2)]), # 3-Day
             np.mean([grid_df['%s'%model]['Day_3']['%s'%(str(i+1))]['errors']['nrmse'] for i in range(day_1-2)]),

             np.mean([grid_df['%s'%model]['Week']['%s'%(str(i+1))]['errors']['mape'] for i in range(day_1-6)]),  # 1-Week
             np.mean([grid_df['%s'%model]['Week']['%s'%(str(i+1))]['errors']['nrmse'] for i in range(day_1-6)])]
            
    return grids_errors

def errors_box(Results, grid, error, day_1=21):
    
    error = error.lower()
    models = Results[list(Results.keys())[0]].keys()
    grid_1 = Results['Grid(%d,%d)'%(grid[0],grid[1])]
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(4, 1, figsize=(20, 20), facecolor='#eaeaf2')
    csfont = {'fontname':'Georgia'}
    hfont = {'fontname':'Calibri'}
    title = 'Box plot for Grid (%d,%d) '%(grid[0],grid[1])
    fig.suptitle(title, y=.97, fontsize=22, color='#525252', **csfont)
    plt.subplots_adjust(top=0.85)

    for i,d_w in enumerate(zip(['Day_1','Day_2','Day_3','Week'],[day_1,day_1-1,day_1-3,day_1-6])):

        temp = pd.DataFrame({'%s'%model:[grid_1['%s'%model]['%s'%d_w[0]]['%s'%(str(i+1))]['errors'][error] 
                                                        for i in range(d_w[1])]
                             for model in models})
        sns.boxplot(data=temp,
                        palette='Set3', 
                        linewidth=1.2, 
                        fliersize=2, 
                        ax=ax[i],
                        flierprops=dict(marker='o', markersize=4))
        ax[i].set_title('%s'%d_w[0])

        ax[i].set_ylabel(error.capitalize(), fontsize=16, color='#525252', **hfont)
        for label in (ax[i].get_xticklabels() + ax[i].get_yticklabels()):
            label.set(fontsize=16, color='#525252', **hfont)
            
    plt.show()
    
def plot_errors(Results:dict, weeks_days:list, models:list, grids:list=['Grid(52,56)']):
    
    if not grids:
        grids = list(Results.keys())
    if not models:
        models = list(Results[grids[0]].keys())

    length = {'Day_1': len(Results[grids[0]][models[0]]['Day_1']) if 'Day_1' in weeks_days else None,
              'Day_2': len(Results[grids[0]][models[0]]['Day_2']) if 'Day_2' in weeks_days else None,
              'Day_3': len(Results[grids[0]][models[0]]['Day_3']) if 'Day_3' in weeks_days else None,
              'Week' : len(Results[grids[0]][models[0]]['Week']) if 'Week' in weeks_days else None}

    for grid in grids:
        print('='*40,grid,'='*40)
        for l in length:
            if length[l]:
                print(' '*35,f'=====  {l}  ======',' '*35)
                fig, axes = plt.subplots(1, len(models), figsize=(20,5))
                for i, model in enumerate(models):
                    errors = pd.DataFrame(index=['%s %s'%(l,i) for i in range(1,length[l]+1,1)])
                    errors['nrmse'] = [Results[grid]['%s'%model]['%s'%(l)]['%d'%(i+1)]['errors']['nrmse'] for i in range(length[l])]
                    errors['mape'] = [Results[grid]['%s'%model]['%s'%(l)]['%d'%(i+1)]['errors']['mape'] for i in range(length[l])]
                    ax = errors.plot(kind='bar', figsize=(20,8), rot='45', ax=[axes if len(models)==1 else axes[i]][0])
                    ax.set_title('%s with %s'%(model, grid))
                plt.show()