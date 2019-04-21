from keras.layers.core import Reshape
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop
from sklearn.utils import shuffle
import imageio
import numpy as np

from model import model_final
from data_load import load

x_train, y_train, x_test, y_test = load()

##### Hyperparameters ##############
####################################
shape = (28, 28, 1)
epoch = 300
batch_size = 64
CheckDir = "sample/"
####################################

model = model_final(shape)

ada = Adadelta(lr=5.0, rho=0.95, epsilon=1e-08, decay=0.001)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)

model.compile(loss="mean_squared_error", optimizer=rms)
model.summary()
# import sys

# sys.exit()

for epoch in range(1, epoch + 1):
    train_X, train_Y = shuffle(x_train, y_train)
    print("Epoch is: %d\n" % epoch)
    print("Number of batches: %d\n" % int(len(train_X) / batch_size))
    num_batches = int(len(train_X) / batch_size)
    for batch in range(num_batches):
        batch_train_X = train_X[
            batch * batch_size : min((batch + 1) * batch_size, len(train_X))
        ]
        batch_train_Y = train_Y[
            batch * batch_size : min((batch + 1) * batch_size, len(train_Y))
        ]
        loss = model.train_on_batch(batch_train_X, batch_train_Y)
        print("epoch_num: %d batch_num: %d loss: %f\n" % (epoch, batch, loss))

    model.save_weights("model.h5")
    if epoch % 5 == 0:
        x_test, y_test = shuffle(x_test, y_test)
        decoded_imgs = model.predict(x_test[:2])
        temp = np.zeros([28, 28 * 3, 3])
        temp[:, :28, :1] = x_test[0, :, :, :1]
        temp[:, 28 : 28 * 2, :1] = y_test[0, :, :, :1]
        temp[:, 28 * 2 :, :1] = decoded_imgs[0, :, :, :1]
        temp[:, :, 1] = temp[:, :, 0]
        temp[:, :, 2] = temp[:, :, 0]
        temp = temp * 255
        imageio.imwrite(CheckDir + str(epoch) + ".jpg", temp)
