                   *******Data preparation and imputaion*********

It’s important for the time series not to have any missing data to be modeled properly 
because the time series data has to be equally spaced as the most of the models assume.
The objective of this task is to try everal methods to know which one is the best in 
the case of mobile network traffic.

 1) Implement the Conv-GAIN paper using TensorFlow framework.
 2) Apply everal methods like:
	a) Conventional methods (e.g. mean, mode, median)
	b) Procedure methods (e.g. Last/Next Valid observation, Seasonal interpolation)
	c) Learnable methods (e.g. GAIN, ConvGAIN, MICE).
 3) Apply the best method (Conv-GAIN) to impute the data.
 4) Visualize the imputed data in an animated figure.


Uses the data from gathering and resampling phase.
 
