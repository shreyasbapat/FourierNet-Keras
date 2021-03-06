from keras.datasets import mnist
import numpy as np
import scipy.fftpack as fp
import scipy.misc
from matplotlib import pyplot as plt


def load():
    (x_train, _), (x_test, _) = mnist.load_data()
    y_train = []
    y_test = []

    im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0), axis=1)
    freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1), axis=0)

    for i in range(len(x_train)):
        freq = im2freq(x_train[i])
        back = freq2im(freq)
        y_train.append(freq)

    for i in range(len(x_test)):
        freq = im2freq(x_test[i])
        back = freq2im(freq)
        y_test.append(freq)
    x_train = np.array(x_train).reshape((60000, 28, 28, -1))
    y_train = np.array(y_train).reshape((60000, 28, 28, -1))
    x_test = np.array(x_test).reshape((10000, 28, 28, -1))
    y_test = np.array(y_test).reshape((10000, 28, 28, -1))

    return x_train, y_train, x_test, y_test
