import random
import numpy as np
np.random.seed(0)
signalOne = np.random.rand(20)
signalTwo = np.random.rand(20)
convolvedTest = np.convolve(signalOne, signalTwo)
print(signalOne)
print(signalTwo)
print(convolvedTest)
