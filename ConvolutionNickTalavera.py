import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft,ifft,fft2,ifft2
import os
from scipy import signal
from scipy import misc
clear = lambda: os.system('cls')
clear()

# The fourier transform allows one to transform the signals
# losslessly to the frequency domain. When in the frequency domain, one can
# multiply two signals to perform convolution.
def convolution(x,y):
    dataSize = np.array(x.shape) # Find the dimensions of input "x"
    kernelSize = np.array(y.shape) # Find the dimensions of input "y"
    padding = dataSize + kernelSize - 1 # Add the dimensions of input "x" and "y"
    # and subtract 1. We calculate the padding so we perform linear convolution
    # instead of circular convolution. Circular convolution shouldn't be used in
    # images because they are not periodic. Using linear convolution instead
    # prevents wrapping the picture or signal on the edges.
    if len(dataSize) == 1 and len(kernelSize) == 1: # Check to see if both "x" and "y" are 1 dimension
        fFreq = fft(x, padding[0]) # Calculate the fourier transform of "x" with padding of length "size"
        gFreq = fft(y, padding[0]) # Calculate the fourier transform of "y" with padding of length "size"
        convolved = fFreq * gFreq # Convolve in the frequency domain by multiplying the two fourier transformed signals
        yPadded = ifft(convolved) # Calculate the inverse fourier transform to restore to the original domain
    elif len(dataSize) == 2 and len(kernelSize) == 2: # Check to see if both "x" and "y" are 2 dimensions
        padding2D = 2 ** np.ceil(np.log2(padding)) # Calculate the 2D padding on each dimension
        xFFT = np.fft.fft2(x, padding2D) # Calculate the fourier transform of "x" with padding of dimension "size"
        yFFT = np.fft.fft2(y, padding2D) # Calculate the fourier transform of "x" with padding of dimension "size"
        convolved = xFFT * yFFT # Convolve in the frequency domain by multiplying the two fourier transformed signals
        yFull = np.fft.ifft2(convolved) # Calculate the inverse fourier transform to restore to the original domain
        vRange = range((padding[1] - dataSize[1])/2, dataSize[1] + (padding[1] - dataSize[1])/2) # Calculate
        # the vertical range for the unpadded signal. If one is convolving two images, the typical desired effect
        # is to get the original dimension of the input data. Otherwise, every convolution would result in larger
        # boundaries along the edges.
        yPadded = yFull[:,vRange] # Unpad the now convolved signals in the vertical range
    hRange = range((padding[0] - dataSize[0])/2, dataSize[0] + (padding[0] - dataSize[0])/2) # Calculate
    # the horizontnal range for the unpadded signal.
    result = yPadded[hRange] # Unpad the now convolved signals in the horizontal range
    y = np.real(result) # Remove imaginary numbers
    return(y)

np.random.seed(0)
signalOne1D = np.random.randn(6)
signalTwo1D = np.random.randn(6)
signalOne2D = np.random.randn(5,4)
signalTwo2D = np.random.randn(5,10)

convolvedActual1D = signal.fftconvolve(signalOne1D, signalTwo1D, mode = 'same')
naive_convolved1DFrequency = convolution(signalOne1D, signalTwo1D);
convolvedActual2D = signal.convolve2d(signalOne2D, signalTwo2D, mode='same')
naive_convolved2D = convolution(signalOne2D, signalTwo2D);

# Unit Tests
# Pass in one empty variable
# Pass in two empty variables
# Pass in incompatible variables
# Pass in two 1D arrays of equal length
# Pass in two 1D arrays of different lengths
# Pass in two 2D arrays
# Pass in a 1D and 2D array
# Pass in a 3D array




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
