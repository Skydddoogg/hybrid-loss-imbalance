import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import clip_ops
from custom_functions.utils import _constant_to_tensor, tf_count

epsilon = backend_config.epsilon

class MeanFalseError(object):

    def __init__(self):
        pass

    def mean_false_error(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        n_positive = math_ops.cast(tf_count(y_true, 1.0), tf.float32)
        n_negative = math_ops.cast(tf_count(y_true, 0.0), tf.float32)

        positive_indices = tf.where(condition = tf.equal(y_true, 1.0))
        negative_indices = tf.where(condition = tf.equal(y_true, 0.0))

        positive_y_true = tf.gather_nd(y_true, positive_indices)
        negative_y_true = tf.gather_nd(y_true, negative_indices)
        positive_y_pred = tf.gather_nd(y_pred, positive_indices)
        negative_y_pred = tf.gather_nd(y_pred, negative_indices)

        FNE = tf.where(tf.equal(n_positive, 0), 0.0, K.mean(math_ops.squared_difference(positive_y_pred, positive_y_true), axis=-1))
        FPE = tf.where(tf.equal(n_negative, 0), 0.0, K.mean(math_ops.squared_difference(negative_y_pred, negative_y_true), axis=-1))

        loss = FNE + FPE

        return loss

    def mean_squared_false_error(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        n_positive = math_ops.cast(tf_count(y_true, 1.0), tf.float32)
        n_negative = math_ops.cast(tf_count(y_true, 0.0), tf.float32)

        positive_indices = tf.where(condition = tf.equal(y_true, 1.0))
        negative_indices = tf.where(condition = tf.equal(y_true, 0.0))

        positive_y_true = tf.gather_nd(y_true, positive_indices)
        negative_y_true = tf.gather_nd(y_true, negative_indices)
        positive_y_pred = tf.gather_nd(y_pred, positive_indices)
        negative_y_pred = tf.gather_nd(y_pred, negative_indices)

        FNE = tf.where(tf.equal(n_positive, 0), 0.0, K.mean(math_ops.squared_difference(positive_y_pred, positive_y_true), axis=-1))
        FPE = tf.where(tf.equal(n_negative, 0), 0.0, K.mean(math_ops.squared_difference(negative_y_pred, negative_y_true), axis=-1))

        loss = (FNE ** 2) + (FPE ** 2)

        return loss

class MeanSquareError(object):

    def __init__(self):
        pass

    def mean_square_error(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        loss = K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)

        return loss

class CrossEntropy(object):

    def __init__(self, alpha = 0.25, label_smoothing = 0):
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def binary_crossentropy(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        label_smoothing = ops.convert_to_tensor(self.label_smoothing, dtype=K.floatx())

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
        pos_loss = -y_true * math_ops.log(p + epsilon())

        # Loss for the negative examples
        neg_loss = -(1 - y_true) * math_ops.log(q + epsilon())

        loss = K.mean(neg_loss + pos_loss)

        return loss

    def balanced_binary_crossentropy(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        label_smoothing = ops.convert_to_tensor(self.label_smoothing, dtype=K.floatx())

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
        pos_loss = -self.alpha * y_true * math_ops.log(p + epsilon())

        # Loss for the negative examples
        neg_loss = -(1 - self.alpha) * (1 - y_true) * math_ops.log(q + epsilon())

        loss = K.mean(neg_loss + pos_loss)

        return loss


class Focal(object):

    def __init__(self, gamma = 2, alpha = 0.25, label_smoothing = 0):
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def focal(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        label_smoothing = ops.convert_to_tensor(self.label_smoothing, dtype=K.floatx())

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
        pos_loss = -(q ** self.gamma) * math_ops.log(p + epsilon())

        # Loss for the negative examples
        neg_loss = -(p ** self.gamma) * math_ops.log(q + epsilon())

        loss = K.mean(y_true * pos_loss + (1 - y_true) * neg_loss)

        return loss

    def balanced_focal(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        label_smoothing = ops.convert_to_tensor(self.label_smoothing, dtype=K.floatx())

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
        pos_loss = -self.alpha * (q ** self.gamma) * math_ops.log(p + epsilon())

        # Loss for the negative examples
        neg_loss = -(1 - self.alpha) * (p ** self.gamma) * math_ops.log(q + epsilon())

        loss = K.mean(y_true * pos_loss + (1 - y_true) * neg_loss)

        return loss

class Hybrid(object):

    def __init__(self, gamma = 2, alpha = 0.25, label_smoothing = 0):
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def hybrid(self, y_true, y_pred):

        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        label_smoothing = ops.convert_to_tensor(self.label_smoothing, dtype=K.floatx())

        def _smooth_labels():
            return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels, lambda: y_true)

        epsilon_ = _constant_to_tensor(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        n_positive = math_ops.cast(tf_count(y_true, 1.0), tf.float32)
        n_negative = math_ops.cast(tf_count(y_true, 0.0), tf.float32)

        positive_indices = tf.where(condition = tf.equal(y_true, 1.0))
        negative_indices = tf.where(condition = tf.equal(y_true, 0.0))

        positive_y_true = tf.gather_nd(y_true, positive_indices)
        negative_y_true = tf.gather_nd(y_true, negative_indices)
        positive_y_pred = tf.gather_nd(y_pred, positive_indices)
        negative_y_pred = tf.gather_nd(y_pred, negative_indices)

        positive_p = positive_y_pred
        negative_p = negative_y_pred

        positive_q = 1 - positive_p
        negative_q = 1 - negative_p

        # Loss for the positive examples
        positive_pos_loss = -(positive_q ** self.gamma) * math_ops.log(positive_p + epsilon())
        negative_pos_loss = -(negative_q ** self.gamma) * math_ops.log(negative_p + epsilon())

        # Loss for the negative examples
        positive_neg_loss = -(positive_p ** self.gamma) * math_ops.log(positive_q + epsilon())
        negative_neg_loss = -(negative_p ** self.gamma) * math_ops.log(negative_q + epsilon())

        positive_loss = tf.where(tf.equal(n_positive, 0), 0.0, K.mean(positive_y_true * positive_pos_loss + (1 - positive_y_true) * positive_neg_loss))
        negative_loss = tf.where(tf.equal(n_negative, 0), 0.0, K.mean(negative_y_true * negative_pos_loss + (1 - negative_y_true) * negative_neg_loss))

        # positive_y_pred = math_ops.cast(tf.where(tf.less_equal(positive_y_pred, 0.5), 0.0, 1.0), tf.float32)
        # negative_y_pred = math_ops.cast(tf.where(tf.less_equal(negative_y_pred, 0.5), 0.0, 1.0), tf.float32)

        # false_neg = math_ops.cast(math_ops.reduce_sum(positive_y_true - positive_y_pred), tf.float32)
        # false_pos = math_ops.cast(math_ops.reduce_sum(negative_y_pred - negative_y_true), tf.float32)

        # positive_cost = tf.where(tf.equal(n_positive, 0), 0.0, math_ops.cast(false_neg / n_positive, tf.float32))
        # negative_cost = tf.where(tf.equal(n_negative, 0), 0.0, math_ops.cast(false_pos / n_negative, tf.float32))

        # loss = positive_cost * positive_loss + negative_cost * negative_loss
        loss = positive_loss + negative_loss

        return loss

    def balanced_hybrid(self, y_true, y_pred):

        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        label_smoothing = ops.convert_to_tensor(self.label_smoothing, dtype=K.floatx())

        def _smooth_labels():
            return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels, lambda: y_true)

        epsilon_ = _constant_to_tensor(epsilon(), y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        n_positive = math_ops.cast(tf_count(y_true, 1.0), tf.float32)
        n_negative = math_ops.cast(tf_count(y_true, 0.0), tf.float32)

        positive_indices = tf.where(condition = tf.equal(y_true, 1.0))
        negative_indices = tf.where(condition = tf.equal(y_true, 0.0))

        positive_y_true = tf.gather_nd(y_true, positive_indices)
        negative_y_true = tf.gather_nd(y_true, negative_indices)
        positive_y_pred = tf.gather_nd(y_pred, positive_indices)
        negative_y_pred = tf.gather_nd(y_pred, negative_indices)

        positive_p = positive_y_pred
        negative_p = negative_y_pred

        positive_q = 1 - positive_p
        negative_q = 1 - negative_p

        # Loss for the positive examples
        positive_pos_loss = -self.alpha * (positive_q ** self.gamma) * math_ops.log(positive_p + epsilon())
        negative_pos_loss = -self.alpha * (negative_q ** self.gamma) * math_ops.log(negative_p + epsilon())

        # Loss for the negative examples
        positive_neg_loss = -(1 - self.alpha) * (positive_p ** self.gamma) * math_ops.log(positive_q + epsilon())
        negative_neg_loss = -(1 - self.alpha) * (negative_p ** self.gamma) * math_ops.log(negative_q + epsilon())

        positive_loss = tf.where(tf.equal(n_positive, 0), 0.0, K.mean(positive_y_true * positive_pos_loss + (1 - positive_y_true) * positive_neg_loss))
        negative_loss = tf.where(tf.equal(n_negative, 0), 0.0, K.mean(negative_y_true * negative_pos_loss + (1 - negative_y_true) * negative_neg_loss))

        # positive_y_pred = math_ops.cast(tf.where(tf.less_equal(positive_y_pred, 0.5), 0.0, 1.0), tf.float32)
        # negative_y_pred = math_ops.cast(tf.where(tf.less_equal(negative_y_pred, 0.5), 0.0, 1.0), tf.float32)

        # false_neg = math_ops.cast(math_ops.reduce_sum(positive_y_true - positive_y_pred), tf.float32)
        # false_pos = math_ops.cast(math_ops.reduce_sum(negative_y_pred - negative_y_true), tf.float32)

        # positive_cost = tf.where(tf.equal(n_positive, 0), 0.0, math_ops.cast(false_neg / n_positive, tf.float32))
        # negative_cost = tf.where(tf.equal(n_negative, 0), 0.0, math_ops.cast(false_pos / n_negative, tf.float32))

        # loss = positive_cost * positive_loss + negative_cost * negative_loss

        loss = positive_loss + negative_loss

        return loss