import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft,ifft,fft2,ifft2
import unittest
from scipy import signal
from scipy import misc
import itertools

def convolution(x,y):
    try:
        x = np.array(x)
        dataSize = np.array(x.shape) # Find the dimensions of input "x"
    except:
        raise ValueError("The input signal is not an array")
    if x.size == 0:
        raise ValueError("The input signal is empty")
    try:
        y = np.array(y)
        kernelSize = np.array(y.shape) # Find the dimensions of input "y"
    except:
        raise ValueError("The kernel signal is not an array")
    if y.size == 0:
        raise ValueError("The kernel signal is empty")
    if len(dataSize) == 1: # Check to see if both "x" and "y" are 1 dimension
        x = np.array([x])
        dataSize = np.array(x.shape) # Find the dimensions of input "x"
    elif len(dataSize) != 2:
        raise ValueError("The input signal not a 1D or 2D array")
    if len(kernelSize) == 1:
        y = np.array([y])
        kernelSize = np.array(y.shape) # Find the dimensions of input "y"
    elif len(kernelSize) != 2:
        raise ValueError("The kernel signal not a 1D or 2D array")
    padding = dataSize + kernelSize - 1 # Add the dimensions of input "x" and "y"
    # and subtract 1. We calculate the padding so we perform linear convolution
    # instead of circular convolution. Circular convolution shouldn't be used in
    # images because they are not periodic. Using linear convolution instead
    # prevents wrapping the picture or signal on the edges.
    padding2D = 2 ** np.ceil(np.log2(padding)).astype(int) # Calculate the 2D padding on each dimension
    xFFT = np.fft.fft2(x, padding2D) # Calculate the fourier transform of "x" with padding of dimension "padding2D"
    yFFT = np.fft.fft2(y, padding2D) # Calculate the fourier transform of "x" with padding of dimension "padding2D"
    convolved = xFFT * yFFT # Convolve in the frequency domain by multiplying the two fourier transformed signals
    yFull = np.fft.ifft2(convolved) # Calculate the inverse fourier transform to restore to the original domain
    vRange = range((padding[1] - dataSize[1])/2, dataSize[1] + (padding[1] - dataSize[1])/2) # Calculate
    # the vertical range for the unpadded signal. If one is convolving two images, the typical desired effect
    # is to get the original dimension of the input data. Otherwise, every convolution would result in larger
    # boundaries along the edges.
    # print(vRange)
    yPadded = yFull[:,vRange] # Unpad the now convolved signals in the vertical range
    hRange = range((padding[0] - dataSize[0])/2, dataSize[0] + (padding[0] - dataSize[0])/2) # Calculate
    # the horizontal range for the unpadded signal.
    result = yPadded[hRange] # Unpad the now convolved signals in the horizontal range
    yOut = np.real(result) # Remove imaginary numbers
    if yOut.shape[0] == 1:
        yOut = yOut[0]
    return(yOut)


# # Unit Tests
class TestConvolution(unittest.TestCase):
    def test_Two_1D_Arrays(self):
        np.random.seed(0)
        numbersToTest = [2,6,7,100]
        for subsetSignal in itertools.permutations(numbersToTest, 2):
            signalOne1D = np.random.randn(subsetSignal[0])
            signalTwo1D = np.random.randn(subsetSignal[1])
            convolvedActual1D = np.around(signal.convolve(signalOne1D, signalTwo1D, mode='same'), decimals = 8)
            naive_convolved1D = np.around(convolution(signalOne1D, signalTwo1D), decimals = 8)
            self.assertTrue(np.array_equal(naive_convolved1D, convolvedActual1D));

    def test_Two_2D_Arrays(self):
        np.random.seed(0)
        numbersToTest = [2,6,7,100]
        for subsetSignalOne in itertools.permutations(numbersToTest, 2):
            for subsetSignalTwo in itertools.permutations(numbersToTest, 2):
                signalOne2D = np.random.randn(subsetSignalOne[0],subsetSignalOne[1])
                signalTwo2D = np.random.randn(subsetSignalTwo[0],subsetSignalTwo[1])
                convolvedActual2D = np.around(signal.convolve2d(signalOne2D, signalTwo2D, mode='same'), decimals = 8)
                naive_convolved2D = np.around(convolution(signalOne2D, signalTwo2D), decimals = 8)
                self.assertTrue(np.array_equal(naive_convolved2D, convolvedActual2D));

    def test_Minimal_Arrays(self):
        np.random.seed(0)
        for subsetSignalOne in itertools.permutations([1,20], 2):
            for subsetSignalTwo in itertools.permutations([1,20], 2):
                signalOne2D = np.random.randn(subsetSignalOne[0],subsetSignalOne[1])
                signalTwo2D = np.random.randn(subsetSignalTwo[0],subsetSignalTwo[1])
                convolvedActual2D = np.around(signal.convolve2d(signalOne2D, signalTwo2D, mode='same'), decimals = 8)
                naive_convolved2D = np.around(convolution(signalOne2D, signalTwo2D), decimals = 8)
                if convolvedActual2D.shape[0] == 1:
                    convolvedActual2D = convolvedActual2D[0]
                self.assertTrue(np.array_equal(naive_convolved2D, convolvedActual2D));

    def test_Different_Dimensions(self):
        np.random.seed(0)
        signalOne1D = np.random.randn(20)
        signalTwo2D = np.random.randn(1,20)
        convolvedActual2D = np.around(signal.convolve2d([signalOne1D], signalTwo2D, mode='same'), decimals = 8)
        naive_convolved2D = np.around(convolution(signalOne1D, signalTwo2D), decimals = 8)
        self.assertTrue(np.array_equal(naive_convolved2D,convolvedActual2D[0]));

    def test_Empty_Arrays(self):
        with self.assertRaises(ValueError):
            np.around(convolution([], []), decimals = 8)
        with self.assertRaises(ValueError):
            np.around(convolution([[]], [[]]), decimals = 8)
        with self.assertRaises(ValueError):
            np.around(convolution([], [1,2,3]), decimals = 8)
        with self.assertRaises(ValueError):
            np.around(convolution([3,2,1], []), decimals = 8)

    def test_High_Dimension_Arrays(self):
        with self.assertRaises(ValueError):
            np.around(convolution(np.arange(30).reshape(2,3,5), np.arange(30).reshape(2,3,5)), decimals = 8)
        with self.assertRaises(ValueError):
            np.around(convolution(np.arange(60).reshape(1,1,3,5), np.arange(30).reshape(1,1,3,5)), decimals = 8)

    def test_Incorrect_Type_Signals(self):
        with self.assertRaises(ValueError):
            np.around(convolution("[1,2,3]", True), decimals = 8)
        with self.assertRaises(ValueError):
            np.around(convolution([1,2,3], "[4,5,6]"), decimals = 8)
        with self.assertRaises(ValueError):
            np.around(convolution("[1,2,3]", [3,2,1]), decimals = 8)

    def test_Tuple_Conversion(self):
        np.random.seed(0)
        numbersToTest = [6,7]
        for subsetSignal in itertools.permutations(numbersToTest, 2):
            signalOne1D = tuple(np.random.randn(subsetSignal[0]))
            signalTwo1D = tuple(np.random.randn(subsetSignal[1]))
            convolvedActual1D = np.around(signal.convolve(signalOne1D, signalTwo1D, mode='same'), decimals = 8)
            naive_convolved1D = np.around(convolution(signalOne1D, signalTwo1D), decimals = 8)
            self.assertTrue(np.array_equal(naive_convolved1D, convolvedActual1D));
        for subsetSignalOne in itertools.permutations(numbersToTest, 2):
            for subsetSignalTwo in itertools.permutations(numbersToTest, 2):
                signalOne2D = tuple(map(tuple, np.random.randn(subsetSignalOne[0],subsetSignalOne[1])))
                signalTwo2D = tuple(map(tuple, np.random.randn(subsetSignalTwo[0],subsetSignalTwo[1])))
                convolvedActual2D = np.around(signal.convolve2d(signalOne2D, signalTwo2D, mode='same'), decimals = 8)
                naive_convolved2D = np.around(convolution(signalOne2D, signalTwo2D), decimals = 8)
                self.assertTrue(np.array_equal(naive_convolved2D, convolvedActual2D));

    def test_List_Conversion(self):
        np.random.seed(0)
        numbersToTest = [6,7]
        for subsetSignal in itertools.permutations(numbersToTest, 2):
            signalOne1D = np.ndarray.tolist(np.random.randn(subsetSignal[0]))
            signalTwo1D = np.ndarray.tolist(np.random.randn(subsetSignal[1]))
            convolvedActual1D = np.around(signal.convolve(signalOne1D, signalTwo1D, mode='same'), decimals = 8)
            naive_convolved1D = np.around(convolution(signalOne1D, signalTwo1D), decimals = 8)
            self.assertTrue(np.array_equal(naive_convolved1D, convolvedActual1D));
        for subsetSignalOne in itertools.permutations(numbersToTest, 2):
            for subsetSignalTwo in itertools.permutations(numbersToTest, 2):
                signalOne2D = np.ndarray.tolist(np.random.randn(subsetSignalOne[0],subsetSignalOne[1]))
                signalTwo2D = np.ndarray.tolist(np.random.randn(subsetSignalTwo[0],subsetSignalTwo[1]))
                convolvedActual2D = np.around(signal.convolve2d(signalOne2D, signalTwo2D, mode='same'), decimals = 8)
                naive_convolved2D = np.around(convolution(signalOne2D, signalTwo2D), decimals = 8)
                self.assertTrue(np.array_equal(naive_convolved2D, convolvedActual2D));

if __name__ == '__main__':
    unittest.main()
