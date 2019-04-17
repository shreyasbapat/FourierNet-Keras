from keras.datasets import mnist
import numpy as np
import scipy.fftpack as fp
import scipy.misc

(x_train, y_train), (x_test, y_test) = mnist.load_data()

im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0), axis=1)
freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1), axis=0)

for i in range(len(x_train)):
    freq = im2freq(x_train[i])
    back = freq2im(freq)
    scipy.misc.imsave("data/" + str(i) + ".jpg", freq)
