import abc
import tensorflow as tf


class AbstractLossFunction(object):
    __metaclass__ = abc.ABCMeta

    # If True, dense prediction results will be passed to the loss function
    is_dense = False

    # If True, randomly sampled predictions will be passed to the loss function
    is_sample_based = False
    # If True, and if is_sample_based is True, predictions will be sampled with replacement
    is_sampled_with_replacement = False

    @abc.abstractmethod
    def connect_loss_graph(self, predictions, interactions):
        """
        """
        pass


class RMSELossGraph(AbstractLossFunction):
    """
    This loss function returns the root mean square error between the predictions and the true interactions.
    Interactions can be any positive or negative values, and this loss function is sensitive to magnitude.
    """
    def connect_loss_graph(self, predictions, interactions):
        return tf.sqrt(tf.reduce_mean(tf.square(interactions - predictions)))


class RMSEDenseLossGraph(AbstractLossFunction):
    """
    This loss function returns the root mean square error between the predictions and the true interactions, including
    all non-interacted values as 0s.
    Interactions can be any positive or negative values, and this loss function is sensitive to magnitude.
    """
    is_dense = True

    def connect_loss_graph(self, predictions, interactions):
        error = tf.sparse_add(interactions, - 1.0 * predictions)
        return tf.sqrt(tf.reduce_mean(tf.square(error)))