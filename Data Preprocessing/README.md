# Preprocessing steps
------
## 1) Anomely detection 
Anomaly detection is the process of discover the event or the points which are unexpected at this position of the dataset or deviates from the normal pattern of the dataset. 
So, the detection of those points very important; because it give us an early step to make the emergency movements to control that un usual change.
- Tokey's Box Plot:
	in this method we depend on the pox plot to determine if the point is outlier or not and not only that it gives us the ability to decide if this outlier is possible or probable outlier point; by calculate the following parameters: 25th percentile (Q1), 75th percentile (Q3), interquartile range (IQR = Q3 – Q1), Lower inner fence: [Q1 – (1.5 * IQR)] , Upper , inner fence: [Q3 + (1.5 * IQR)], Lower outer fence: [Q1 – (3 * IQR)], Upper outer fence: [Q3 + (3 * IQR)].
	![Tokes output](images/r6.png)
- Isolation Forest Method:
	Isolation Forest build using the decision trees which depend on the points that go deeper into the tree are not anomalies and points which go short distance have big probability to be anomalies, and it is unsupervised learning model which used without labeled data.[24]
	![Isolation Ouput](images/r8.png)
- LSTM Autoencoders Method:
	In this method we will depend on the detection using the forecasting by Deep Learning algorithms. In the forecasting methods we depend on predict the next point with the addition of some noise and make comparison of this point and the true point at this timestamp by finding the difference between the two points then add threshold finally find the anomalies by compare the difference of the two points with this threshold (we used the Mean absolute error MAE).[23]
	![autoencoder output](images/r9.png)
- Seasonal-Trend Decomposition Method:
	Signal decomposition aims to analysis our signal to its main three components Seasonal, trend and the residual (S, T, R). Seasonal is the signal component which contain the most rapidly pattern which occurs regular every certain time. Trend contain the general shape of the data over the whole dataset and finally the residual is the rest of the signal after extract the seasonal and trend of it, it is in somehow a random part over the signal which indicate it.[25]
	![Decompse output](images/r10.png)

## 2) Missing data imputation models
We have several methods to impute time series data
- Conventional methods:
  - Ignore or deletion.
  - Mean imputation.
  - Mode imputation.
  - Median imputation.
- Imputation procedure:
  - Last valid observation forward.
  - Next valid observation forward.
  - Interpolation.
- Learnable methods:
  - KNN algorithm.
  - AutoRegressive.
  - Genitic algorithm.
  - MICE algorithm.
  - Least square SVM.
  - GAIN.
  - Conv-GAIN.
