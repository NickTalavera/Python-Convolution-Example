## convolution.py
## Written by Nick Talavera on February 13, 2017

import unittest
import random
import itertools
import numpy as np
from numpy.random import randn, seed
from numpy.fft import fft,ifft,fft2,ifft2
from scipy import signal

def convolution(x,y):
## Returns a convolved array from an input array and a kernel array with the shape
## of the original input array.
##
## Inputs:
## x = a 1D or 2D input array that may be written in as an array, tuple or list and must be >= 1 in length
## y = a 1D or 2D kernel array that may be written in as an array, tuple or list and must be >= 1 in length

    # Test to see if the arrays are valid types. If so, get the shape of each.
    try:
        x = np.array(x) # Convert "x" into a numpy array
        dataSize = np.array(x.shape) # Find the dimensions of input "x"
    except:
        raise ValueError("The input signal is not an array")
    try:
        y = np.array(y) # Convert "y" into a numpy array
        kernelSize = np.array(y.shape) # Find the dimensions of kernel "y"
    except:
        raise ValueError("The kernel signal is not an array")

    # Test to see if the arrays are empty.
    if x.size == 0:
        raise ValueError("The input signal is empty")
    if y.size == 0:
        raise ValueError("The kernel signal is empty")

    # Test to see if the arrays are 1D or 2D. If 1D, change to 2D to be processed by fft2.
    if len(dataSize) == 1: # Check to see if both "x" and "y" are 1 dimension
        x = np.array([x])  # Convert "x" into 2d
        dataSize = np.array(x.shape) # Find the dimensions of input "x"
    elif len(dataSize) != 2:
        raise ValueError("The input signal not a 1D or 2D array")
    if len(kernelSize) == 1:
        y = np.array([y]) # Convert "y" into 2d
        kernelSize = np.array(y.shape) # Find the dimensions of kernel "y"
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
    convolved = xFFT * yFFT # Convolve in the frequency domain by multiplying the two fourier transformed arrays
    convolvedFull = np.fft.ifft2(convolved) # Calculate the inverse fourier transform to restore to the original domain
    vRange = range((padding[1] - dataSize[1])/2, dataSize[1] + (padding[1] - dataSize[1])/2) # Calculate
    # the vertical range for the unpadded signal. If one is convolving two images, the typical desired effect
    # is to get the original dimension of the input data. Otherwise, every convolution would result in padded
    # boundaries along the convolved array.
    convolvedPadded = convolvedFull[:,vRange] # Unpad the now convolved arrays in the vertical range
    hRange = range((padding[0] - dataSize[0])/2, dataSize[0] + (padding[0] - dataSize[0])/2) # Calculate
    # the horizontal range for the unpadded arrays
    result = convolvedPadded[hRange] # Unpad the now convolved arrays in the horizontal range
    convolvedOut = np.real(result) # Remove imaginary numbers
    if convolvedOut.shape[0] == 1: # If the array can be converted to 1D, remove the outer bracket
        convolvedOut = convolvedOut[0] # Remove the outer bracket
    return(convolvedOut) # Return the convolved array


class TestConvolution(unittest.TestCase):
    ## Unit tests for the convolution function
    seed(0) # Set the random seed to 0 for consistency

    def test_Two_1D_Arrays(self):
        ## Test if two 1D arrays are properly convolved with eachother in the convolution function.
        numbersToTest = [2,6,7,100] # Possible dimensions of a signal to be permuted and tested
        for subsetSignal in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest"
            signalOne1D = randn(subsetSignal[0]) # Generate a signal
            signalTwo1D = randn(subsetSignal[1]) # Generate another signal
            trueConvolved1D = np.around(signal.convolve(signalOne1D, signalTwo1D, mode='same'), decimals = 10) # Try the default convolution function
            testConvolved1D = np.around(convolution(signalOne1D, signalTwo1D), decimals = 10) # Try the custom convolution function
            self.assertTrue(np.array_equal(testConvolved1D, trueConvolved1D)) # Success if arrays are equal

    def test_Two_2D_Arrays(self):
        ## Test if two 2D arrays are properly convolved with eachother in the convolution function.
        numbersToTest = [2,6,7,100] # Possible dimensions of a signal to be permuted and tested
        for subsetSignalOne in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalOne
            for subsetSignalTwo in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalTwo
                signalOne2D = randn(subsetSignalOne[0],subsetSignalOne[1]) # Generate a signal
                signalTwo2D = randn(subsetSignalTwo[0],subsetSignalTwo[1]) # Generate another signal
                trueConvolved2D = np.around(signal.convolve2d(signalOne2D, signalTwo2D, mode='same'), decimals = 10) # Try the default convolution function
                testConvolved2D = np.around(convolution(signalOne2D, signalTwo2D), decimals = 10) # Try the custom convolution function
                self.assertTrue(np.array_equal(testConvolved2D, trueConvolved2D)) # Success if arrays are equal

    def test_Minimal_Arrays(self):
        ## Test if 1X# or #X1 arrays can be convolved with eachother in the convolution function.
        for subsetSignalOne in itertools.permutations([1,20], 2): # For all permutations to create 1X# or #X1 arrays
            for subsetSignalTwo in itertools.permutations([1,20], 2): # For all permutations 1X# or #X1 arrays for a second signal
                signalOne2D = randn(subsetSignalOne[0],subsetSignalOne[1]) # Generate a signal
                signalTwo2D = randn(subsetSignalTwo[0],subsetSignalTwo[1]) # Generate another signal
                trueConvolved2D = np.around(signal.convolve2d(signalOne2D, signalTwo2D, mode='same'), decimals = 10) # Try the default convolution function
                testConvolved2D = np.around(convolution(signalOne2D, signalTwo2D), decimals = 10) # Try the custom convolution function
                if trueConvolved2D.shape[0] == 1: # If the true array should be 1D for comparison, remove outer bracket
                    trueConvolved2D = trueConvolved2D[0]
                self.assertTrue(np.array_equal(testConvolved2D, trueConvolved2D)) # Success if arrays are equal

    def test_Different_Dimensions(self):
        ## Test if 1D and 2D functions can be convolved with eachother in the convolution function.
        signalOne1D = randn(20) # Generate a 1D signal
        signalTwo2D = randn(4,20) # Generate a 2D signal
        trueConvolved2D = np.around(signal.convolve2d([signalOne1D], signalTwo2D, mode='same'), decimals = 10) # Try the default convolution function
        testConvolved2D = np.around(convolution(signalOne1D, signalTwo2D), decimals = 10) # Try the custom convolution function
        self.assertTrue(np.array_equal(testConvolved2D,trueConvolved2D[0])) # Success if arrays are equal

    def test_Empty_Arrays(self):
        ## Test if empty arrays throw errors in the convolution function.
        with self.assertRaises(ValueError): # Success if an error is thrown for two empty 1D arrays
            np.around(convolution([], []), decimals = 10)
        with self.assertRaises(ValueError): # Success if an error is thrown for two empty 2D arrays
            np.around(convolution([[]], [[]]), decimals = 10)
        with self.assertRaises(ValueError): # Success if an error is thrown for one empty 1D input
            np.around(convolution([], [1,2,3]), decimals = 10)
        with self.assertRaises(ValueError): # Success if an error is thrown for one empty 1D kernel
            np.around(convolution([3,2,1], []), decimals = 10)

    def test_High_Dimension_Arrays(self):
        ## Test if high dimension (>2D) arrays throw errors in the convolution function.
        with self.assertRaises(ValueError): # Success if an error is thrown for a 3D array
            np.around(convolution(np.arange(30).reshape(2,3,5), np.arange(30).reshape(2,3,5)), decimals = 10)
        with self.assertRaises(ValueError): # Success if an error is thrown for a 4D array
            np.around(convolution(np.arange(60).reshape(1,1,3,5), np.arange(30).reshape(1,1,3,5)), decimals = 10)

    def test_Incorrect_Type_arrays(self):
        ## Test if invalid types throw errors in the convolution function.
        with self.assertRaises(ValueError): # Success if an error is thrown for two incorrect types
            np.around(convolution("[1,2,3]", True), decimals = 10)
        with self.assertRaises(ValueError): # Success if an error is thrown for one incorrect kernel type
            np.around(convolution([1,2,3], "[4,5,6]"), decimals = 10)
        with self.assertRaises(ValueError): # Success if an error is thrown incorrect input type
            np.around(convolution("[1,2,3]", [3,2,1]), decimals = 10)

    def test_Tuple_Conversion(self):
        ## Test if tuples can be used in the convolution function.
        numbersToTest = [6,7] # Possible dimensions of a signal to be permuted and tested
        for subsetSignal in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest"
            signalOne1D = tuple(randn(subsetSignal[0])) # Generate a 1D signal as a tuple
            signalTwo1D = tuple(randn(subsetSignal[1])) # Generate another 1D signal as a tuple
            trueConvolved1D = np.around(signal.convolve(signalOne1D, signalTwo1D, mode='same'), decimals = 10) # Try the default convolution function
            testConvolved1D = np.around(convolution(signalOne1D, signalTwo1D), decimals = 10) # Try the custom convolution function
            self.assertTrue(np.array_equal(testConvolved1D, trueConvolved1D)) # Success if arrays are equal
        for subsetSignalOne in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalOne
            for subsetSignalTwo in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalTwo
                signalOne2D = tuple(map(tuple, randn(subsetSignalOne[0],subsetSignalOne[1]))) # Generate a 2D signal as a list
                signalTwo2D = tuple(map(tuple, randn(subsetSignalTwo[0],subsetSignalTwo[1]))) # Generate another 2D signal as a list
                trueConvolved2D = np.around(signal.convolve2d(signalOne2D, signalTwo2D, mode='same'), decimals = 10) # Try the default convolution function
                testConvolved2D = np.around(convolution(signalOne2D, signalTwo2D), decimals = 10) # Try the custom convolution function
                self.assertTrue(np.array_equal(testConvolved2D, trueConvolved2D)) # Success if arrays are equal

    def test_List_Conversion(self):
        ## Test if lists can be used in the convolution function.
        numbersToTest = [6,7] # Possible dimensions of a signal to be permuted and tested
        for subsetSignal in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest"
            signalOne1D = np.ndarray.tolist(randn(subsetSignal[0])) # Generate a 1D signal as a list
            signalTwo1D = np.ndarray.tolist(randn(subsetSignal[1])) # Generate another 1D signal as a list
            trueConvolved1D = np.around(signal.convolve(signalOne1D, signalTwo1D, mode='same'), decimals = 10) # Try the default convolution function
            testConvolved1D = np.around(convolution(signalOne1D, signalTwo1D), decimals = 10) # Try the custom convolution function
            self.assertTrue(np.array_equal(testConvolved1D, trueConvolved1D)) # Success if arrays are equal
        for subsetSignalOne in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalOne
            for subsetSignalTwo in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalTwo
                signalOne2D = np.ndarray.tolist(randn(subsetSignalOne[0],subsetSignalOne[1])) # Generate a 2D signal as a list
                signalTwo2D = np.ndarray.tolist(randn(subsetSignalTwo[0],subsetSignalTwo[1])) # Generate another 2D signal as a list
                trueConvolved2D = np.around(signal.convolve2d(signalOne2D, signalTwo2D, mode='same'), decimals = 10) # Try the default convolution function
                testConvolved2D = np.around(convolution(signalOne2D, signalTwo2D), decimals = 10) # Try the custom convolution function
                self.assertTrue(np.array_equal(testConvolved2D, trueConvolved2D)) # Success if arrays are equal

if __name__ == '__main__':
    unittest.main() # Run the unittest
