from tensorflow.keras.layers import Conv2D, MaxPool2D, concatenate
from tensorflow import keras

# Xavier uniform initializer.
kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)


def inception_module(X, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
    """
    Inception layer as describe at the Google paper “Going deeper with convolutions”

    :param X: Input layer
    :param filters_1x1: Conv2D 1x1 filter size
    :param filters_3x3_reduce: Conv2D 3x3 reduce filter size
    :param filters_3x3: Conv2D 3x3 filter size
    :param filters_5x5_reduce: Conv2D 3x3 reduce filter size
    :param filters_5x5: Conv2D 5x5 filter size
    :param filters_pool_proj: Conv2D projection filter size
    :param name: Block name
    :return output: Concatenated output
    """
    # 1x1 Conv block
    conv_1x1 = Conv2D(filters_1x1, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(X)

    # 3x3 Conv block
    conv_3x3 = Conv2D(filters_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu',
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    conv_3x3 = Conv2D(filters_3x3, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_3x3)

    # 5x5 Conv block
    conv_5x5 = Conv2D(filters_5x5_reduce, kernel_size=(1, 1), padding='same', activation='relu',
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    conv_5x5 = Conv2D(filters_5x5, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_5x5)

    # Poolproj block
    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(X)
    pool_proj = Conv2D(filters_pool_proj, kernel_size=(1, 1), padding='same', activation='relu',
                       kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output
