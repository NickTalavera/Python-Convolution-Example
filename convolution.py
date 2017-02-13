import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft,ifft,fft2,ifft2
import unittest
from scipy import signal
from scipy import misc
import itertools

def convolution(x,y):
    # try:
    dataSize = np.array(x.shape) # Find the dimensions of input "x"
    kernelSize = np.array(y.shape) # Find the dimensions of input "y"
    padding = dataSize + kernelSize - 1 # Add the dimensions of input "x" and "y"
    # and subtract 1. We calculate the padding so we perform linear convolution
    # instead of circular convolution. Circular convolution shouldn't be used in
    # images because they are not periodic. Using linear convolution instead
    # prevents wrapping the picture or signal on the edges.
    # print(padding)
    # print(dataSize)
    if len(dataSize) == 1 and len(kernelSize) == 1: # Check to see if both "x" and "y" are 1 dimension
        fFreq = fft(x, padding[0]) # Calculate the fourier transform of "x" with padding of length "size"
        gFreq = fft(y, padding[0]) # Calculate the fourier transform of "y" with padding of length "size"
        convolved = fFreq * gFreq # Convolve in the frequency domain by multiplying the two fourier transformed signals
        yPadded = ifft(convolved) # Calculate the inverse fourier transform to restore to the original domain
    elif len(dataSize) == 2 and len(kernelSize) == 2: # Check to see if both "x" and "y" are 2 dimensions
        padding2D = 2 ** np.ceil(np.log2(padding)).astype(int) # Calculate the 2D padding on each dimension
        xFFT = np.fft.fft2(x, padding2D) # Calculate the fourier transform of "x" with padding of dimension "size"
        yFFT = np.fft.fft2(y, padding2D) # Calculate the fourier transform of "x" with padding of dimension "size"
        convolved = xFFT * yFFT # Convolve in the frequency domain by multiplying the two fourier transformed signals
        yFull = np.fft.ifft2(convolved) # Calculate the inverse fourier transform to restore to the original domain
        vRange = range((padding[1] - dataSize[1])/2, dataSize[1] + (padding[1] - dataSize[1])/2) # Calculate
        # the vertical range for the unpadded signal. If one is convolving two images, the typical desired effect
        # is to get the original dimension of the input data. Otherwise, every convolution would result in larger
        # boundaries along the edges.
        # print(vRange)
        yPadded = yFull[:,vRange] # Unpad the now convolved signals in the vertical range
    hRange = range((padding[0] - dataSize[0])/2, dataSize[0] + (padding[0] - dataSize[0])/2) # Calculate
    # the horizontnal range for the unpadded signal.
    result = yPadded[hRange] # Unpad the now convolved signals in the horizontal range
    y = np.real(result) # Remove imaginary numbers
    return(y)
    # except:
        # raise ValueError("Arrays must have the same size")

# # Unit Tests
class TestConvolution(unittest.TestCase):
# # Pass in one empty variable
# # Pass in two empty variables
# # Pass in incompatible variables
# # Pass in a 1D and 2D array
# # Pass in a 3D array
    def test_Two_1D_Arrays(self):
        np.random.seed(0)
        numbersToTest = [2,6,7,100]
        for subsetSignal in itertools.permutations(numbersToTest, 2):
            signalOne1D = np.random.randn(subsetSignal[0])
            signalTwo1D = np.random.randn(subsetSignal[1])
            convolvedActual1D = np.around(signal.convolve(signalOne1D, signalTwo1D, mode='same'), decimals = 8)
            naive_convolved1D = np.around(convolution(signalOne1D, signalTwo1D), decimals = 8)
            self.assertTrue(np.array_equal(convolvedActual1D, naive_convolved1D));

    def test_Two_2D_Arrays(self):
        np.random.seed(0)
        numbersToTest = [2,6,7,100]
        for subsetSignalOne in itertools.permutations(numbersToTest, 2):
            for subsetSignalTwo in itertools.permutations(numbersToTest, 2):
                signalOne2D = np.random.randn(subsetSignalOne[0],subsetSignalOne[1])
                signalTwo2D = np.random.randn(subsetSignalTwo[0],subsetSignalTwo[1])
                convolvedActual2D = np.around(signal.convolve2d(signalOne2D, signalTwo2D, mode='same'), decimals = 8)
                naive_convolved2D = np.around(convolution(signalOne2D, signalTwo2D), decimals = 8)
                self.assertTrue(np.array_equal(convolvedActual2D, naive_convolved2D));

    def test_Minimal_Arrays(self):
        np.random.seed(0)
        for subsetSignalOne in itertools.permutations([1,20], 2):
            for subsetSignalTwo in itertools.permutations([1,20], 2):
                signalOne2D = np.random.randn(subsetSignalOne[0],subsetSignalOne[1])
                signalTwo2D = np.random.randn(subsetSignalTwo[0],subsetSignalTwo[1])
                convolvedActual2D = np.around(signal.convolve2d(signalOne2D, signalTwo2D, mode='same'), decimals = 8)
                naive_convolved2D = np.around(convolution(signalOne2D, signalTwo2D), decimals = 8)
                self.assertTrue(np.array_equal(convolvedActual2D, naive_convolved2D));

    def test_Different_Dimensions(self):
        np.random.seed(0)
        signalOne2D = np.array([np.random.randn(20)])
        signalTwo2D = np.random.randn(1,20)
        print(signalOne2D)
        print(signalTwo2D)
        convolvedActual2D = np.around(signal.convolve2d(signalOne2D, signalTwo2D, mode='same'), decimals = 8)
        naive_convolved2D = np.around(convolution(signalOne2D, signalTwo2D), decimals = 8)
        self.assertTrue(np.array_equal(convolvedActual2D, naive_convolved2D));


    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
