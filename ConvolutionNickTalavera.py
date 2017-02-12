import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft,ifft,fft2,ifft2
import os
from scipy import signal
from scipy import misc
clear = lambda: os.system('cls')
clear()

# Convolution is easier when the signal is transofrmed into the frequency domain.
# This is 1D convolution in the time domain where you "slide" a kernel, g, along
# the signal, f, while adding the product of the signals together at each offset.

# An example of 1D linear convolution in the time domain.
def convolve1DTimeDomain(f, g):
    dataSize = len(f)
    kernelSize = len(g) - 1
    outlen = dataSize + kernelSize
    y = np.zeros(outlen)
    for t in range(0, outlen):
        for x in range(max(t - kernelSize, 0), min(t + 1, dataSize)):
            y[t] += f[x] * g[t - x]
    return(y)

# The fourier transform allows one to transform the signals
# losslessly to the frequency domain. When in the frequency domain, one can
# multiply two signals to perform convolution. To return to the original domain,
# an inverse fourier transform must be performed.

# An example of 1D convolution in the frequency domain.
# The ffts are performed with zero padding for linear convolution instead of
# circular convolution. If we used circular convolution, one side of the image
# would wrap around to the other side of the image.
def convolve1DFrequencyDomain(f, g):
    fFreq = fft(f, len(f) + len(g) - 1)
    gFreq = fft(g, len(f) + len(g) - 1)
    cv = fFreq*gFreq
    y = ifft(cv)
    return(y)


def naive_convolved2D(x,y):
    dataSize = np.array(x.shape)
    kernelSize = np.array(y.shape)
    size = dataSize + kernelSize - 1
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    xFFT = np.fft.fft2(x, fsize)
    yFFT = np.fft.fft2(y, fsize)
    fsliceH = range((size[0] - dataSize[0])/2, dataSize[0] + (size[0] - dataSize[0])/2)
    fsliceV = range((size[1] - dataSize[1])/2, dataSize[1] + (size[1] - dataSize[1])/2)
    result = np.fft.ifft2(xFFT * yFFT)
    result = result[fsliceH][:,fsliceV]
    print(dataSize)
    print(kernelSize)
    print(size)
    print(range((size[0] - dataSize[0])/2, dataSize[0] + (size[0] - dataSize[0])/2))
    print("fsliceH = " + str(fsliceH))
    print("fsliceV = " + str(fsliceV))
    print(fsize)
    # print(new_x)
    # print(result.shape)
    print("result = " + str(result))
    return np.real(result)

# def naive_convolved2D(x,y):
#     s1 = np.array(x.shape)
#     s2 = np.array(y.shape)
#     size = s1 + s2 - 1
#     print(size)
#     fsize = 2 ** np.ceil(np.log2(size)).astype(int)
#     print(np.log2(size))
#     # fsize = 2**np.ceil(np.sqrt(size))
#     print(fsize)
#     fslice = tuple([slice(0, int(sz)) for sz in size])
#     print(fslice)
#     new_x = np.fft.fft2(x , fsize)
#     new_y = np.fft.fft2(y , fsize)
#     result = np.fft.ifft2(new_x*new_y)[fslice].copy()
#     print(result.shape)
#     return np.real(result)

np.random.seed(0)
signalOne1D = np.random.randn(6)
signalTwo1D = np.random.randn(6)
convolvedActual1D = signal.fftconvolve(signalOne1D, signalTwo1D)
naive_convolved1DTime = convolve1DTimeDomain(signalOne1D, signalTwo1D);
naive_convolved1DFrequency = convolve1DFrequencyDomain(signalOne1D, signalTwo1D);


signalOne2D = np.random.randn(5,6)
signalTwo2D = np.random.randn(5,7)
# convolvedActual2D = signal.convolve2d(signalOne2D, signalTwo2D, boundary='symm', mode='same')
convolvedActual2D = signal.convolve2d(signalOne2D, signalTwo2D, mode='same')
# print(convolvedActual2D)
naive_convolved2D = naive_convolved2D(signalOne2D, signalTwo2D);
# print(convolvedActual2D)
# print(naive_convolved2D)
# print(convolvedActual2D.shape)
# print(naive_convolved2D.shape)
# diff2d = np.subtract(convolvedActual2D,naive_convolved2D)
# print(diff2d)

plt.plot(convolvedActual2D)
plt.title('Numpy 2D Convolution')
plt.show()
plt.plot(naive_convolved2D)
plt.title('Nick\'s 2D Convolution')
plt.show()
# plt.plot(naive_convolved1DFrequency)
# plt.title('Nick\'s 1D Frequency Domain Convolution')
# plt.show()
# plt.plot(naive_convolved1DTime)
# plt.title('Nick\'s 1D Time Domain Convolution')
# plt.show()
# plt.plot(convolvedActual1D)
# plt.title('Numpy 1D Convolution')
# plt.show()
