import os
import unittest
from convolution import convolution
import numpy as np
from scipy import signal
from scipy import misc

np.random.seed(0)
signalOne1D = np.random.randn(6)
signalTwo1D = np.random.randn(6)
signalOne2D = np.random.randn(6,4)
signalTwo2D = np.random.randn(6,10)

convolvedActual1D = signal.fftconvolve(signalOne1D, signalTwo1D, mode = 'same')
naive_convolved1DFrequency = convolution(signalOne1D, signalTwo1D);
convolvedActual2D = signal.convolve2d(signalOne2D, signalTwo2D, mode='same')
naive_convolved2D = convolution(signalOne2D, signalTwo2D);

# Unit Tests
# Pass in one empty variable
# convolution([],[])
# Pass in two empty variables
# Pass in incompatible variables
# Pass in two 1D arrays of equal length
# Pass in two 1D arrays of different lengths
# Pass in two 2D arrays
# Pass in a 1D and 2D array
# Pass in a 3D array
# import sys
# def main(n):
#     res = fact(n)
#     print(res)

# if __name__ == '__main__':
# naive_convolved2D = convolution(signalOne2D, signalTwo2D);




plt.plot(convolvedActual2D)
plt.title('Numpy 2D Convolution')
plt.show()
plt.plot(naive_convolved2D)
plt.title('Nick\'s 2D Convolution')
plt.show()
plt.plot(naive_convolved1DFrequency)
plt.title('Nick\'s 1D Convolution')
plt.show()
plt.plot(convolvedActual1D)
plt.title('Numpy 1D Convolution')
plt.show()
