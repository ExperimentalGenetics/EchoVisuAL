import tensorflow as tf
from keras.layers import Layer


class CustomDropout(Layer):
    def __init__(self, rate, **kwargs):
        """
        Constructor for the Custom Dropout-Layer that is enabled during training and inference
        :param rate: dropout rate 0.1 - 0.2
        :param kwargs:kwargs
        """
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, **kwargs):
        """
        Cll Method for dropout layer
        :param inputs: input
        :param kwargs: kwargs
        :return: tensorflow dropout layer with specified dropout rate
        """
        return tf.nn.dropout(inputs, rate=self.rate)
