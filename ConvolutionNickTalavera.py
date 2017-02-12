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
def convolve1DFrequencyDomain(x, y):
    dataSize = len(x)
    kernelSize = len(y)
    size = dataSize + kernelSize - 1
    # size = [size[0], 1]
    fFreq = fft(x, size)
    gFreq = fft(y, size)
    cv = fFreq*gFreq
    y = ifft(cv)
    return(y)


def naive_convolved2D(x,y):
    print(x)
    x = np.array(x)
    dataSize = np.array(x.shape)
    kernelSize = np.array(y.shape)
    size = dataSize + kernelSize - 1
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    xFFT = np.fft.fft2(x, fsize)
    yFFT = np.fft.fft2(y, fsize)
    convolved = xFFT * yFFT
    result = np.fft.ifft2(convolved)
    fsliceH = range((size[0] - dataSize[0])/2, dataSize[0] + (size[0] - dataSize[0])/2)
    fsliceV = range((size[1] - dataSize[1])/2, dataSize[1] + (size[1] - dataSize[1])/2)
    result = result[fsliceH][:,fsliceV]
    return np.real(result)

def combined(x,y):
    dataSize = np.array(x.shape)
    kernelSize = np.array(y.shape)
    if len(dataSize) == 1 and len(kernelSize) == 1:
        size = dataSize + kernelSize - 1
        fFreq = fft(x, size[0])
        gFreq = fft(y, size[0])
        convolved = fFreq * gFreq
        yFull = ifft(convolved)
    else:
        x = np.array(x)
        dataSize = np.array(x.shape)
        kernelSize = np.array(y.shape)
        size = dataSize + kernelSize - 1
        fsize = 2 ** np.ceil(np.log2(size)).astype(int)
        xFFT = np.fft.fft2(x, fsize)
        yFFT = np.fft.fft2(y, fsize)
        convolved = xFFT * yFFT
        result = np.fft.ifft2(convolved)
        vRange = range((size[1] - dataSize[1])/2, dataSize[1] + (size[1] - dataSize[1])/2)
        yFull = result[:,vRange]
    hRange = range((size[0] - dataSize[0])/2, dataSize[0] + (size[0] - dataSize[0])/2)
    result = yFull[hRange]
    y = np.real(result)
    return(y)

np.random.seed(0)
signalOne1D = np.random.randn(6)
signalTwo1D = np.random.randn(6)
convolvedActual1D = signal.fftconvolve(signalOne1D, signalTwo1D, mode = 'same')
naive_convolved1DFrequency = combined(signalOne1D, signalTwo1D);


signalOne2D = np.random.randn(5,4)
signalTwo2D = np.random.randn(5,10)
convolvedActual2D = signal.convolve2d(signalOne2D, signalTwo2D, mode='same')
naive_convolved2D = combined(signalOne2D, signalTwo2D);


plt.plot(convolvedActual2D)
plt.title('Numpy 2D Convolution')
plt.show()
plt.plot(naive_convolved2D)
plt.title('Nick\'s 2D Convolution')
plt.show()
plt.plot(naive_convolved1DFrequency)
plt.title('Nick\'s 1D Frequency Domain Convolution')
plt.show()
# plt.plot(naive_convolved1DTime)
# plt.title('Nick\'s 1D Time Domain Convolution')
# plt.show()
plt.plot(convolvedActual1D)
plt.title('Numpy 1D Convolution')
plt.show()
