import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name="F1", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)

        def _zero_wt_init(name):
            return self.add_weight(name, shape=[], initializer="zeros", dtype=self.dtype)

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred > .5
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        def _count_non_zero(val):
            non_zeros = tf.math.count_nonzero(val, axis=None)
            return tf.cast(non_zeros, self.dtype)

        self.true_positives.assign_add(_count_non_zero(y_pred * y_true))
        self.false_positives.assign_add(_count_non_zero(y_pred * (y_true - 1)))
        self.false_negatives.assign_add(_count_non_zero((y_pred - 1) * y_true))

    def result(self):
        precision = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_positives)
        recall = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        return(tf.math.divide_no_nan(2 * precision * recall, precision + recall))

    def reset_states(self):
        self.true_positives.assign(tf.zeros([], tf.float32))
        self.false_positives.assign(tf.zeros([], tf.float32))
        self.false_negatives.assign(tf.zeros([], tf.float32))