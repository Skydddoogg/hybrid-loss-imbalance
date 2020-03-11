import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend_config
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import clip_ops

epsilon = backend_config.epsilon

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def _constant_to_tensor(x, dtype):
  return constant_op.constant(x, dtype=dtype)

def mean_false_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    n_positive = tf_count(y_true, 1.0)
    n_negative = tf_count(y_true, 0.0)

    positive_indices = tf.where(condition = tf.equal(y_true, 1.0))
    negative_indices = tf.where(condition = tf.equal(y_true, 0.0))

    positive_y_true = tf.gather_nd(y_true, positive_indices)
    negative_y_true = tf.gather_nd(y_true, negative_indices)
    positive_y_pred = tf.gather_nd(y_pred, positive_indices)
    negative_y_pred = tf.gather_nd(y_pred, negative_indices)

    FNE = math_ops.multiply(math_ops.cast(tf.where(tf.equal(n_positive, 0), 0.0, math_ops.cast(1 / n_positive, tf.float32)), tf.float32), math_ops.reduce_sum(math_ops.multiply(1 / 2, math_ops.square(positive_y_pred - positive_y_true))))
    FPE = math_ops.multiply(math_ops.cast(tf.where(tf.equal(n_negative, 0), 0.0, math_ops.cast(1 / n_negative, tf.float32)), tf.float32), math_ops.reduce_sum(math_ops.multiply(1 / 2, math_ops.square(negative_y_pred - negative_y_true))))

    loss = FNE + FPE

    return loss

def mean_square_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    n_positive = tf_count(y_true, 1.0)
    n_negative = tf_count(y_true, 0.0)

    n_sample = n_negative + n_positive

    loss = math_ops.multiply(math_ops.cast(1 / n_sample, tf.float32), math_ops.reduce_sum(math_ops.multiply(1 / 2, math_ops.square(y_pred - y_true))))

    return loss

def focal(y_true, y_pred, gamma = 2, label_smoothing=0):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    label_smoothing = ops.convert_to_tensor(label_smoothing, dtype=K.floatx())

    def _smooth_labels():
        return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels, lambda: y_true)

    epsilon_ = _constant_to_tensor(epsilon(), y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

    # Predicted prob for positive class
    p = y_pred

    # Predicted prob for negative class
    q = 1 - p

    # Loss for the positive examples
    pos_loss = -(q ** gamma) * math_ops.log(p)

    # Loss for the negative examples
    neg_loss = -(p ** gamma) * math_ops.log(q)

    loss = K.mean(y_true * pos_loss + (1 - y_true) * neg_loss)

    return loss

def binary_crossentropy(y_true, y_pred, label_smoothing=0):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    label_smoothing = ops.convert_to_tensor(label_smoothing, dtype=K.floatx())

    def _smooth_labels():
        return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels, lambda: y_true)

    epsilon_ = _constant_to_tensor(epsilon(), y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

    # Predicted prob for positive class
    p = y_pred

    # Predicted prob for negative class
    q = 1 - p

    # Loss for the positive examples
    pos_loss = -y_true * math_ops.log(p)

    # Loss for the negative examples
    neg_loss = -(1 - y_true) * math_ops.log(q)

    loss = K.mean(neg_loss + pos_loss)

    return loss

def hybrid_mfe_fl(y_true, y_pred):

    mfe = mean_false_error(y_true, y_pred)
    fl = focal(y_true, y_pred)

    loss = K.mean(mfe + fl)

    return loss