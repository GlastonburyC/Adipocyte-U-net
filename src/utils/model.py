from keras.callbacks import Callback
from keras import backend as K
from math import ceil
from time import ctime
import logging
import tensorflow as tf

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    return K.mean(jac)

def mean_diff(y_true, y_pred):
    return K.mean(y_pred) - K.mean(y_true)

def act_mean(y_true, y_pred):
    return K.mean(y_pred)

def act_min(y_true, y_pred):
    return K.min(y_pred)

def act_max(y_true, y_pred):
    return K.max(y_pred)

def act_std(y_true, y_pred):
    return K.std(y_pred)

def tru_pos(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmax(y_true, axis=axis)
    ypam = K.argmax(y_pred, axis=axis)
    prod = ytam * ypam
    return K.sum(prod)

def fls_pos(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmax(y_true, axis=axis)
    ypam = K.argmax(y_pred, axis=axis)
    diff = ypam - ytam
    return K.sum(K.clip(diff, 0, 1))

def tru_neg(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmin(y_true, axis=axis)
    ypam = K.argmin(y_pred, axis=axis)
    prod = ytam * ypam
    return K.sum(prod)

def fls_neg(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmin(y_true, axis=axis)
    ypam = K.argmin(y_pred, axis=axis)
    diff = ypam - ytam
    return K.sum(K.clip(diff, 0, 1))

def precision_onehot(y_true, y_pred):
    '''Custom implementation of the keras precision metric to work with
    one-hot encoded outputs.'''
    axis = K.ndim(y_true) - 1
    yt_flat = K.cast(K.argmax(y_true, axis=axis), 'float32')
    yp_flat = K.cast(K.argmax(y_pred, axis=axis), 'float32')
    true_positives = K.sum(K.round(K.clip(yt_flat * yp_flat, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(yp_flat, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_onehot(y_true, y_pred):
    '''Custom implementation of the keras recall metric to work with
        one-hot encoded outputs.'''
    axis = K.ndim(y_true) - 1
    yt_flat = K.cast(K.argmax(y_true, axis=axis), 'float32')
    yp_flat = K.cast(K.argmax(y_pred, axis=axis), 'float32')
    true_positives = K.sum(K.round(K.clip(yt_flat * yp_flat, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(yt_flat, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fmeasure_onehot(y_true, y_pred):
    '''Custom implementation of the keras fmeasure metric to work with
    one-hot encoded outputs.'''
    p = precision_onehot(y_true, y_pred)
    r = recall_onehot(y_true, y_pred)
    return 2 * (p * r) / (p + r + K.epsilon())

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -1.0 * dice_coef(y_true, y_pred)

def weighted_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    kernel_size = 21
    y_true=tf.expand_dims(y_true, 0)
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - weighted_dice_coeff(y_true, y_pred, weight)
    return loss

def weighted_dice_coeff(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return score

def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    kernel_size = 21
    y_true=tf.expand_dims(y_true, 0)
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + (1 - weighted_dice_coeff(y_true, y_pred, weight))
    return loss

class KerasHistoryPlotCallback(Callback):

    def on_train_begin(self, logs={}):
        self.logs = {}

    def on_epoch_end(self, epoch, logs={}):

        if hasattr(self, 'file_name'):
            import matplotlib
            matplotlib.use('agg')

        import matplotlib.pyplot as plt

        if len(self.logs) == 0:
            self.logs = {key:[] for key in logs.keys()}

        for key,val in logs.items():
            self.logs[key].append(val)

        nb_metrics = len([k for k in self.logs.keys() if not k.startswith('val')])
        nb_col = 6
        nb_row = int(ceil(nb_metrics * 1.0 / nb_col))
        fig, axs = plt.subplots(nb_row, nb_col, figsize=(min(nb_col * 3, 12), 3 * nb_row))
        for idx, ax in enumerate(fig.axes):
            if idx >= len(self.logs):
                ax.axis('off')
                continue
            key = sorted(self.logs.keys())[idx]
            if key.startswith('val_'):
                continue
            ax.set_title(key)
            ax.plot(self.logs[key], label='TR')
            val_key = 'val_%s' % key
            if val_key in self.logs:
                ax.plot(self.logs[val_key], label='VL')
            ax.legend()

        plt.suptitle('Epoch %d: %s' % (epoch, ctime()), y=1.10)
        plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)

        if hasattr(self, 'file_name'):
            plt.savefig(self.file_name)
        else:
            plt.show()

class KerasSimpleLoggerCallback(Callback):

    def on_train_begin(self, logs={}):
        self.prev_logs = None
        return
    def on_epoch_end(self, epoch, logs={}):

        logger = logging.getLogger(__name__)

        if self.prev_logs == None:
            for key,val in logs.items():
                logger.info('%15s: %.5lf' % (key,val))
        else:
            for key,val in logs.items():
                diff = val - self.prev_logs[key]
                logger.info('%20s: %15.4lf %5s %15.4lf' % \
                    (key, val, '+' if diff > 0 else '-', abs(diff)))

        self.prev_logs = logs
