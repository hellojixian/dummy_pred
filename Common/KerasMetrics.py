import keras.backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mean_signed_deviation(y_true, y_pred):
    return K.sum(K.mean(y_pred - y_true))


def max_abs_error(y_true, y_pred):
    return K.max(K.abs(y_pred - y_true))
