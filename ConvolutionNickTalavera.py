import random
import matplotlib.pyplot as plt
import numpy as np
import os
clear = lambda: os.system('cls')
clear()

def convolver(f, g):
    dataSize = len(f)
    kernelSize = len(g) - 1
    outlen = dataSize + kernelSize
    y = np.zeros(outlen)
    for t in range(0, outlen):
        print("")
        print("t = " + str(t))
        print(range(max(t - kernelSize, 0), min(t + 1, dataSize)))
        for x in range(max(t - kernelSize, 0), min(t + 1, dataSize)):
            print("t = " + str(t))
            print("x = " + str(x))
            print("t - x = " + str(t - x))
            y[t] += f[x] * g[t - x]
    return(y)

np.random.seed(0)
signalOne = np.random.randn(6)
signalTwo = np.random.randn(6)
# print(signalOne)
# print(signalTwo)
convolvedActual = np.convolve(signalOne, signalTwo)
naive_convolved = convolver(signalOne, signalTwo);


# naive_convolved = convolve(signalOne, signalTwo)
print("convolvedActual")
print(convolvedActual)
print("naive_convolved")
print(naive_convolved)
plt.plot(convolvedActual)
plt.ylabel('some numbers')
plt.show()
plt.plot(naive_convolved)
plt.ylabel('some numbers')
plt.show()
