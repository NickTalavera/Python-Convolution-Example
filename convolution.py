## convolution.py
## This script includes a function for discrete linear convolution, unit tests
## for the convolution function, instructions for applying time domain linear
## filters, and edge detector and blur filters.
##
## Written by Nick Talavera on February 13, 2017
##
## To run, type:
## python convolution.py
##
import unittest
import random
import itertools
import numpy as np
from numpy.random import randn, seed
from numpy.fft import fft,ifft,fft2,ifft2
from scipy import signal

def convolution(x,y):
## Returns the discrete linear convolution of an input array and a kernel array.
## The output will be the same dimensions of the input array.
##
## Inputs:
## x = a 1D or 2D time/spatial domain input matrix that may be written in as a
##      rectangular array, tuple or list and must be >= 1 in length. This is
##      typically the image or signal.
## y = a 1D or 2D time/spatial domain kernel matrix that may be written in as a
##      rectangular array, tuple or list and must be >= 1 in length. This is
##      typically a second signal/image or a filter.
##
## Example:
## convolution([3, 2, 1], [[3,2,1],[3,2,1]])
##
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

    shouldConvertTo1D = False
    # Test to see if the arrays are 1D or greater than 2 dimensions. If 1D,
    # change to 2D to be compatible with fft2.
    if len(dataSize) == 1: # Check to see if both "x" and "y" are 1 dimension
        x = np.array([x])  # Convert "x" into 2d
        shouldConvertTo1D = True
        dataSize = np.array(x.shape) # Find the dimensions of input "x"
    elif len(dataSize) != 2:
        raise ValueError("The input signal not a 1D or 2D array")
    if len(kernelSize) == 1:
        y = np.array([y]) # Convert "y" into 2d
        kernelSize = np.array(y.shape) # Find the dimensions of kernel "y"
    elif len(kernelSize) != 2:
        raise ValueError("The kernel signal not a 1D or 2D array")

    # We calculate the padding so we can perform linear convolution instead of
    # circular convolution. Circular convolution shouldn't be used in images
    # because images are not periodic. Using `linear convolution instead prevents
    # wrapping the image or signal on the edges.
    padding = dataSize + kernelSize - 1 # Add the dimension of input "x" and "y" and subtract 1
    padding2D = 2 ** np.ceil(np.log2(padding)).astype(int) # Find the nearest 2^x power of the padding. The FFT function requires ints.

    # Frequency domain convolution can be more computationally efficient than
    # doing time domain convolution for large arrays. Tests for failures in FFT
    # calculations.
    try:
        xFFT = np.fft.fft2(x, padding2D) # Calculate the fourier transform of "x" with padding of dimension "padding2D"
    except:
        raise ValueError("The input is not usable. Check to be sure it is a rectangle and has the proper dimensions.")
    try:
        yFFT = np.fft.fft2(y, padding2D) # Calculate the fourier transform of "y" with padding of dimension "padding2D"
    except:
        raise ValueError("The kernel is not usable. Check to be sure it is a rectangle and has the proper dimensions.")
    convolved = xFFT * yFFT # Convolve in the frequency domain by multiplying the two fourier transformed arrays
    convolvedFull = np.fft.ifft2(convolved) # Calculate the inverse fourier transform to restore to the original domain

    # If one is convolving two images, the typical desired effect is to get the
    # original dimension of the input data. Otherwise, every convolution would
    # result in padded boundaries along the convolved array.
    vRange = range((padding[1] - dataSize[1]) / 2, dataSize[1] + (padding[1] - dataSize[1])/2) # Calculate
    # the vertical range for the unpadded signal.
    hRange = range((padding[0] - dataSize[0]) / 2, dataSize[0] + (padding[0] - dataSize[0])/2) # Calculate
    # the horizontal range for the unpadded arrays
    result = convolvedFull[hRange][:, vRange] # Unpad the now convolved arrays

    # Clean up the final output.
    convolvedOut = np.real(result) # Remove imaginary numbers
    if convolvedOut.shape[0] == 1 and shouldConvertTo1D == True: # If the array can be converted back to 1D, remove the outer bracket
        convolvedOut = convolvedOut[0] # Remove the outer bracket
    return(convolvedOut) # Return the convolved array


class TestConvolution(unittest.TestCase):
    ## Unit tests for the convolution function
    seed(0) # Set the random seed to 0 for consistency

    def test_Two_1D_Arrays(self):
        ## Test if two 1D arrays are properly convolved with eachother in the convolution function.
        numbersToTest = [2, 6, 7, 15, 100] # Possible dimensions of a signal to be permuted and tested
        for subsetSignal in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest"
            signalOne1D = randn(subsetSignal[0]) # Generate a signal
            signalTwo1D = randn(subsetSignal[1]) # Generate another signal
            trueConvolved1D = signal.convolve(signalOne1D, signalTwo1D, mode='same') # Try the default convolution function
            testConvolved1D = convolution(signalOne1D, signalTwo1D) # Try the custom convolution function
            self.assertTrue(np.array_equal(np.around(testConvolved1D, decimals = 8), np.around(trueConvolved1D, decimals = 8))) # Success if arrays are equal

    def test_Two_2D_Arrays(self):
        ## Test if two 2D arrays are properly convolved with eachother in the convolution function.
        numbersToTest = [2, 6, 7, 15, 100] # Possible dimensions of a signal to be permuted and tested
        for subsetSignalOne in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalOne
            for subsetSignalTwo in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalTwo
                signalOne2D = randn(subsetSignalOne[0], subsetSignalOne[1]) # Generate a signal
                signalTwo2D = randn(subsetSignalTwo[0], subsetSignalTwo[1]) # Generate another signal
                trueConvolved2D = signal.convolve2d(signalOne2D, signalTwo2D, mode='same') # Try the default convolution function
                testConvolved2D = convolution(signalOne2D, signalTwo2D) # Try the custom convolution function
                self.assertTrue(np.array_equal(np.around(testConvolved2D, decimals = 8), np.around(trueConvolved2D, decimals = 8))) # Success if arrays are equal

    def test_Minimal_Arrays(self):
        ## Test if 1X# or #X1 arrays can be convolved with eachother in the convolution function.
        for subsetSignalOne in itertools.permutations([1, 20], 2): # For all permutations to create 1X# or #X1 arrays
            for subsetSignalTwo in itertools.permutations([1, 20], 2): # For all permutations 1X# or #X1 arrays for a second signal
                signalOne2D = randn(subsetSignalOne[0], subsetSignalOne[1]) # Generate a signal
                signalTwo2D = randn(subsetSignalTwo[0], subsetSignalTwo[1]) # Generate another signal
                trueConvolved2D = signal.convolve2d(signalOne2D, signalTwo2D, mode='same') # Try the default convolution function
                testConvolved2D = convolution(signalOne2D, signalTwo2D) # Try the custom convolution function
                self.assertTrue(np.array_equal(np.around(testConvolved2D, decimals = 8), np.around(trueConvolved2D, decimals = 8))) # Success if arrays are equal

    def test_Different_Dimensions(self):
        ## Test if 1D and 2D functions can be convolved with eachother in the convolution function.
        signalOne1D = randn(20) # Generate a 1D signal
        signalTwo2D = randn(4, 20) # Generate a 2D signal
        trueConvolved2D = signal.convolve2d([signalOne1D], signalTwo2D, mode='same') # Try the default convolution function
        testConvolved2D = convolution(signalOne1D, signalTwo2D) # Try the custom convolution function
        self.assertTrue(np.array_equal(np.around(testConvolved2D, decimals = 8), np.around(trueConvolved2D[0], decimals = 8))) # Success if arrays are equal

    def test_Did_Convert_To_1D(self):
        ## Test if 1D depth 1D arrays are kept 1D in the output and 2D depth 1D arrays are kept 2D in the output.
        signalOne1D = randn(17) # Generate a 1D signal
        signalTwo1D = randn(4) # Generate another 1D signal
        test1D1D = convolution(signalOne1D, signalTwo1D).ndim # 1D input, 1D kernel
        self.assertTrue(np.array_equal(test1D1D, 1)) # Success if outputs 1D
        test1D2D = convolution(signalOne1D, [signalTwo1D]).ndim # 1D input, 2D kernel
        self.assertTrue(np.array_equal(test1D2D, 1)) # Success if outputs 1D
        test2D1D = convolution([signalOne1D], signalTwo1D).ndim # 2D input, 1D kernel
        self.assertTrue(np.array_equal(test2D1D, 2)) # Success if outputs 2D
        test2D2D = convolution([signalOne1D], [signalTwo1D]).ndim # 2D input, 2D kernel
        self.assertTrue(np.array_equal(test2D2D, 2)) # Success if outputs 2D


    def test_Empty_Variables(self):
        ## Test if empty variables throw errors in the convolution function.
        with self.assertRaises(ValueError): # Success if an error is thrown for two empty 1D arrays
            convolution([], [])
        with self.assertRaises(ValueError): # Success if an error is thrown for two empty 2D arrays
            convolution([[]], [[]])
        with self.assertRaises(ValueError): # Success if an error is thrown for two empty 3D arrays
            convolution([[[]]], [[[]]])
        with self.assertRaises(ValueError): # Success if an error is thrown for two Nones
            convolution(None, None)
        with self.assertRaises(ValueError): # Success if an error is thrown for one empty 1D input
            convolution([], [1, 2, 3])
        with self.assertRaises(ValueError): # Success if an error is thrown for one empty 1D kernel
            convolution([3, 2, 1], [])

    def test_High_Dimension_Arrays(self):
        ## Test if >2 dimension arrays throw errors in the convolution function.
        with self.assertRaises(ValueError): # Success if an error is thrown for a 3D array
            convolution(np.arange(30).reshape(2, 3, 5), np.arange(30).reshape(2, 3, 5))
        with self.assertRaises(ValueError): # Success if an error is thrown for a 4D array
            convolution(np.arange(60).reshape(1, 1, 3, 5), np.arange(30).reshape(1, 1, 3, 5))

    def test_Incorrect_Type_arrays(self):
        ## Test if invalid types throw errors in the convolution function.
        with self.assertRaises(ValueError): # Success if an error is thrown for two incorrect types
            convolution("[1, 2, 3]", True)
        with self.assertRaises(ValueError): # Success if an error is thrown for one incorrect kernel type
            convolution([1, 2, 3], "[4, 5, 6]")
        with self.assertRaises(ValueError): # Success if an error is thrown incorrect input type
            convolution("[1, 2, 3]", [3, 2, 1])
        with self.assertRaises(ValueError): # Success if an error is thrown for an incorrect input type in 1D array form
            convolution([True, False, "2"], [3, 2, 1])
        with self.assertRaises(ValueError): # Success if an error is thrown for an incorrect input type in 2D array form
            convolution([[True, False, "2"], [True, False, "2"]],[3, 2, 1])
        with self.assertRaises(ValueError): # Success if an error is thrown for an incorrect kernel type in 1D array form
            convolution([3, 2, 1], [True, False, "2"])
        with self.assertRaises(ValueError): # Success if an error is thrown for an incorrect kernel type in 2D array form
            convolution([3, 2, 1], [[True, False, "2"], [True, False, "2"]])

    def test_Tuple_Conversion(self):
        ## Test if tuples can be used in the convolution function.
        numbersToTest = [6, 7] # Possible dimensions of a signal to be permuted and tested
        #Test 1D tuples
        for subsetSignal in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest"
            signalOne1D = tuple(randn(subsetSignal[0])) # Generate a 1D signal as a tuple
            signalTwo1D = tuple(randn(subsetSignal[1])) # Generate another 1D signal as a tuple
            trueConvolved1D = signal.convolve(signalOne1D, signalTwo1D, mode='same') # Try the default convolution function
            testConvolved1D = convolution(signalOne1D, signalTwo1D) # Try the custom convolution function
            self.assertTrue(np.array_equal(np.around(testConvolved1D, decimals = 8), np.around(trueConvolved1D, decimals = 8))) # Success if arrays are equal
        #Test 2D tuples
        for subsetSignalOne in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalOne
            for subsetSignalTwo in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalTwo
                signalOne2D = tuple(map(tuple, randn(subsetSignalOne[0], subsetSignalOne[1]))) # Generate a 2D signal as a list
                signalTwo2D = tuple(map(tuple, randn(subsetSignalTwo[0], subsetSignalTwo[1]))) # Generate another 2D signal as a list
                trueConvolved2D = signal.convolve2d(signalOne2D, signalTwo2D, mode='same') # Try the default convolution function
                testConvolved2D = convolution(signalOne2D, signalTwo2D) # Try the custom convolution function
                self.assertTrue(np.array_equal(np.around(testConvolved2D, decimals = 8), np.around(trueConvolved2D, decimals = 8))) # Success if arrays are equal

    def test_List_Conversion(self):
        ## Test if lists can be used in the convolution function.
        numbersToTest = [6, 7] # Possible dimensions of a signal to be permuted and tested
        #Test 1D lists
        for subsetSignal in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest"
            signalOne1D = np.ndarray.tolist(randn(subsetSignal[0])) # Generate a 1D signal as a list
            signalTwo1D = np.ndarray.tolist(randn(subsetSignal[1])) # Generate another 1D signal as a list
            trueConvolved1D = signal.convolve(signalOne1D, signalTwo1D, mode='same') # Try the default convolution function
            testConvolved1D = convolution(signalOne1D, signalTwo1D) # Try the custom convolution function
            self.assertTrue(np.array_equal(np.around(testConvolved1D, decimals = 8), np.around(trueConvolved1D, decimals = 8))) # Success if arrays are equal
        #Test 2D lists
        for subsetSignalOne in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalOne
            for subsetSignalTwo in itertools.permutations(numbersToTest, 2): # For all permutations of "numbersToTest" for signalTwo
                signalOne2D = np.ndarray.tolist(randn(subsetSignalOne[0], subsetSignalOne[1])) # Generate a 2D signal as a list
                signalTwo2D = np.ndarray.tolist(randn(subsetSignalTwo[0], subsetSignalTwo[1])) # Generate another 2D signal as a list
                trueConvolved2D = signal.convolve2d(signalOne2D, signalTwo2D, mode='same') # Try the default convolution function
                testConvolved2D = convolution(signalOne2D, signalTwo2D) # Try the custom convolution function
                self.assertTrue(np.array_equal(np.around(testConvolved2D, decimals = 8), np.around(trueConvolved2D, decimals = 8))) # Success if arrays are equal

    def test_Nonrectangular_Lists(self):
        ## Test if nonrectangular lists are rejected.
        normalList= randn(6, 11) # Generate a 2D array
        nonrectangularList = np.array([[1,2,3,4], [1,2,3], [1,2]]) # A nonrectangular list
        with self.assertRaises(ValueError): # Success if a value error is raised for a nonrectangular input
            convolution(nonrectangularList, normalList)
        with self.assertRaises(ValueError): # Success if a value error is raised for two nonrectangular kernel
            convolution(normalList, nonrectangularList)
        with self.assertRaises(ValueError): # Success if a value error is raised for two nonrectangular matrices
            convolution(nonrectangularList, nonrectangularList)

if __name__ == '__main__':
    unittest.main() # Run the unittest

## Applying a linear filter to time series data
##
## To apply a linear filter to time series, design an 1D array in the frequency
## domain with the desired response. Note that when designing the filter, the
## frequency should be in designed with pi radians per sample. Then apply an
## inverse fourier transform on signal to return the filter to the time domain
## for use in the convolution function.
##
##
## Building an edge detector or blur filter for a 2D image
##
## Convolve the image with the edge detector filter:
## [[0 1 0]
## [1 -4 1]
## [0 1 0]]
##
## The following filter is an outline filter which can highlight edges
## slightly more dramatically:
## [[1 1 1]
## [1 -8 1]
## [1 1 1]]
##
## Blur filter
## Convolve the image with a gaussian filter array. You can use a different sigma
## and kernel size for different blur effects. This is one example blur filter:
## [[1/16 1/8 1/6]
## [1/8 1/4 1/8]
## [1/16 1/8 1/16]]
