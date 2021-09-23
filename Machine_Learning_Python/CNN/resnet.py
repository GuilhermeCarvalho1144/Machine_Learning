from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2


def res_identity(X, filters):
    '''
    resnet block where dimensions doesn't change
    The skip connection is simple the identity connection
    We will have 3 blocks and then the output will be add
    :param X: Input layer
    :param filters: filters dimensions
    :return X: Output layer
    '''
    X_skip = X
    f1, f2 = filters

    # 1st block
    X = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(X)
    X = BatchNormalization()(X)
    X = Activation(activations.relu)(X)

    # 2nd block = bottleneck
    X = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(X)
    X = BatchNormalization()(X)
    X = Activation(activations.relu)(X)

    # 3nd block
    X = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(X)
    X = BatchNormalization()(X)
    X = Add()([X, X_skip])
    X = Activation(activations.relu)(X)

    return X


def res_conv(X, s, filters):
    '''
    Here the input size changes, when it goes via conv blocks
    So the skip connections uses a projection (conv layer) matrix
    :param X: Input layer
    :param filters: filters dimensions
    :return X: Output layer
    '''
    X_skip = X
    f1, f2 = filters

    # first block
    X = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(X)
    # when s = 2 then it is like downsizing the feature map
    X = BatchNormalization()(X)
    X = Activation(activations.relu)(X)

    # second block
    X = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(X)
    X = BatchNormalization()(X)
    X = Activation(activations.relu)(X)

    # third block
    X = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(X)
    X = BatchNormalization()(X)

    # shortcut
    X_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(X_skip)
    X_skip = BatchNormalization()(X_skip)

    # add
    X = Add()([X, X_skip])
    X = Activation(activations.relu)(X)

    return X
