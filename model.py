from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Model


def model_final(shape):
    """
    Model for autoencoder

    Parameters
    ----------
    shape : tuple

    Returns
    -------
    autoencoder : ~keras.models.Model

    """
    inputs = Input(shape=shape)
    enco = Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    enco = BatchNormalization()(enco)
    enco = Conv2D(16, (3, 3), activation="relu", padding="same")(enco)
    enco = BatchNormalization()(enco)
    enco = MaxPooling2D(pool_size=(2, 2))(enco)

    enco = Conv2D(32, (3, 3), activation="relu", padding="same")(enco)
    enco = BatchNormalization()(enco)
    enco = Conv2D(32, (3, 3), activation="relu", padding="same")(enco)
    enco = BatchNormalization()(enco)
    enco = MaxPooling2D(pool_size=(2, 2))(enco)

    # enco = Conv2D(64, (3, 3), activation='relu', padding='same')(enco)
    # enco = BatchNormalization()(enco)
    # enco = Conv2D(64, (3, 3), activation='relu', padding='same')(enco)
    # enco = BatchNormalization()(enco)
    # enco = MaxPooling2D(pool_size=(2, 2))(enco)
    #
    # enco = Conv2D(128, (3, 3), activation='relu', padding='same')(enco)
    # enco = BatchNormalization()(enco)
    # enco = Conv2D(128, (3, 3), activation='relu', padding='same')(enco)
    # enco = BatchNormalization()(enco)

    # deco = UpSampling2D((2,2))(enco)
    # deco = Conv2D(128, (3, 3), activation='relu', padding='same')(deco)
    # deco = BatchNormalization()(deco)
    # deco = Conv2D(128, (3, 3), activation='relu', padding='same')(deco)
    # deco = BatchNormalization()(deco)

    # deco = UpSampling2D((2,2))(enco)
    # deco = Conv2D(64, (3, 3), activation='relu', padding='same')(deco)
    # deco = BatchNormalization()(deco)
    # deco = Conv2D(64, (3, 3), activation='relu', padding='same')(deco)
    # deco = BatchNormalization()(deco)

    deco = UpSampling2D((2, 2))(enco)
    deco = Conv2D(32, (3, 3), activation="relu", padding="same")(deco)
    deco = BatchNormalization()(deco)
    deco = Conv2D(32, (3, 3), activation="relu", padding="same")(deco)
    deco = BatchNormalization()(deco)

    deco = UpSampling2D((2, 2))(deco)
    deco = Conv2D(16, (3, 3), activation="relu", padding="same")(deco)
    deco = BatchNormalization()(deco)
    deco = Conv2D(16, (3, 3), activation="relu", padding="same")(deco)
    deco = BatchNormalization()(deco)

    decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(deco)

    autoencoder = Model(inputs, decoded)

    return autoencoder
