<h1>Millan Data Imputation And Forecasting</h1>
<h2>The objective of this project is:</h2>
  <h3>1) Gather and preprocess the <a href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV">Telecommunications Millan Data</a></h3><br>
    <ul>
      <li> Read the txt-file as csv.
      <li> Groupe the data by time and grid.
      <li> Concatinate all the data in one DataFrame.
      <li> Convevrt the full data into a numpy array, where each sample is a (100, 100) matrix.
      <li> Each matrix is the measurements for all grids at a spicific time step.
      <li> Each value in the matrix is a measurement for one grid at a this time step.
    </ul>
  <h3>2) Apply many missing data imputation methods to the data</h3><br>
  <ul>
    <li> Missing data imputation methodes :
      <ul>
      <li> Conventional methods:
        <ul>
          <li> Ignore or deletion.
          <li> Mean imputaion.
          <li> Mode imputaion.
          <li> Median imputaion.
        </ul>
      <li> Imputation procedure:
        <ul>
          <li> Last valid observation forward.
          <li> Next valid observation forward.
          <li> Interpolation.
        </ul>
      <li> Learnable methods:
        <ul>
          <li> KNN algorithm.
          <li> AutoRegressive.
          <li> Genitic algorithm.
          <li> MICE algorithm.
          <li> Least square SVM.
          <li> GAIN.
          <li> Conv-GAIN.
        </ul>
      </ul>
    </ul>
  <h2>3) Time series Forecasting: </h2><br>
<ul>
- Statistical methods:
  
  - Common Approaches:
      - Trend, Seasonal, Residual Decompositions:
          - Seasonal Extraction in ARIMA Time Series (SEATS).
          - Seasonal and Trend decomposition using Loess (STL). 
          - Exponential smoothing:
              - Single Exponential Smoothing, or SES, for univariate data without trend or seasonality.
              - Double Exponential Smoothing for univariate data with support for trends.
              - Triple Exponential Smoothing, or Holt-Winters Exponential Smoothing, with support for both trends and seasonality.(TES)
      - Autoregressive Models (AR).
      - Moving Average Models (MA).

  - Box–Jenkins Approaches: 
      - ARIMA.
      - SARIMA.

- Machine Learning Methods
    - KNN.
    - SVR.
    - LR.
    - ElasticNet.
    - Lasso.
    
- Deep Learning Methods:
    - MLP.
    - CNN.
    - LSTM.
  
 </ul>
      </ul>
