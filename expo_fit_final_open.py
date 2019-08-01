# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from scipy.optimize import least_squares
import warnings
import matplotlib.pyplot as plt

x = np.loadtxt('time-open-zeroed.txt', unpack=True)
y = np.loadtxt('fluorescence-open.txt', unpack=True)

def func(x, F, a, k, b, m):
    return F + a*(1-np.exp(-k*x)) + b*(1-np.exp(-m*x))

def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") #do not print wranings
    val = func(x, *parameterTuple)
    return np.sum((y - val)**2.0)

def generate_Initial_Parameters():
    # min and max used for bounds

    parameterBounds = [(0,100), (-100, 2), (0, 100), (-0.01, 2), (0, 1)]

    #"seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=5)
    return result.x


#generate initial parameters values
geneticParameters = generate_Initial_Parameters()
#curve fit the test data
fittedParameters, pcov = curve_fit(func, x, y, geneticParameters)
print('Parameters', fittedParameters)
modelPredictions = func(x, *fittedParameters)
absError = modelPredictions - y
SE = np.square(absError) #square errors
MSE = np.mean(SE) #mean square errors
RMSE = np.sqrt(MSE) #root mean squared error, RMSE
Rsquared = 1.0 - (np.var(absError)/np.var(y))
print('RMSE:', RMSE)
print('R-squared:', Rsquared)

print()

def compareplt(data1,data2):

     #Plot latência e jitter
      plt.rcParams['font.size']=12
      plt.rcParams['figure.figsize']=[18,6]
      lat = plt.plot(data1, data2, color='blue',label='data')


      plt.legend(loc='best')
      #plt.title('')
      plt.xlabel('Time(s)')
      plt.ylabel('Normalized Fluorescence')
      plt.show(block=False)

#graphics output section
def ModelAndScatterPlot(graphWidth, graphHeight):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = f.add_subplot(111)

    # first the raw data as a scatter plot
    axes.plot(x, y,  '.')

    # create data for the fitted equation plot
    xModel = np.linspace(min(x), max(x))
    yModel = func(xModel, *fittedParameters)

    # now the model as a line plot
    axes.plot(xModel, yModel)

    axes.set_xlabel('Time(s)') # X axis data label
    axes.set_ylabel('Normalized Fluorescence') # Y axis data label

    plt.show()
    plt.close('all') # clean up after using pyplot


graphWidth = 1000
graphHeight = 600
ModelAndScatterPlot(graphWidth, graphHeight)
