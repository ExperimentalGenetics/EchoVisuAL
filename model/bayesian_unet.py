from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Conv2DTranspose

from model.custom_dropout import CustomDropout


class BayesianUnet(Model):

    def __init__(self, mode, dropout_rate, batch_size, *args, **kwargs):
        """
        Constructor to initialize the BayesianUNet
        :param mode:  train|inference|train_scratch
        :param dropout_rate: 0.1 - 0.2
        :param batch_size: number of gpu replicas
        :param args: args
        :param kwargs: kwargs
        """
        super().__init__(*args, **kwargs)

        self.mode = mode
        self.dropout_rate = dropout_rate
        self.image_shape = (1024, 1024, 3)
        self.batch_size = batch_size

    def convolutional_block(self, _input, name, filters, kernel_size=3):
        """
        Convolutional block, 2 Conv2D filters
        :param _input: input for first convolutional layer
        :param name: name of the convolutional block
        :param filters: filters used for both convolutions
        :param kernel_size: kernel size used for both convolutions
        :return: output of the convolutional block
        """
        x = Conv2D(filters, kernel_size, activation='relu', padding='same', name=name + "a",
                   batch_size=self.batch_size)(_input)
        x = Conv2D(filters, kernel_size, activation='relu', padding='same', name=name + "b",
                   batch_size=self.batch_size)(x)
        return x

    def encoder_block(self, _input, num, filters):
        """
        Encoder block - Convolutional block, max-pool, dropout
        :param _input: input for the convolutional block
        :param num: number of the level - used for naming the layers
        :param filters: filters used for both convolutions
        :return: output of the convolutional block, output of the dropout
        """
        contracting_conv = self.convolutional_block(_input, "con_conv_" + num, filters)
        max_pool = MaxPooling2D(pool_size=(2, 2), name="max_pool_" + num, batch_size=self.batch_size)(contracting_conv)
        dropout = CustomDropout(self.dropout_rate, name="encoder_dropout_" + num)(max_pool)
        return contracting_conv, dropout

    def decoder_block(self, input_from_underneath, input_from_same_level, filters, num):
        """
        Decoder block - Up-Sampling, Attention-Gate(optional), concat, convolutional block, dropout
        :param input_from_underneath: input layer from a lower level of the unet
        :param input_from_same_level: input layer from the same level of the unet
        :param filters: filters used for both convolutions and the attention gate
        :param num: number of the level - used for naming the layers
        :return: output of the dropout layer
        """
        up_sampling = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same',
                                      name="up_sampling_" + num)(input_from_underneath)
        concat = Concatenate(name="Concat_" + num, batch_size=self.batch_size)([up_sampling, input_from_same_level])
        expanding_conv = self.convolutional_block(concat, "expanding_conv_" + num, filters)
        dropout = CustomDropout(self.dropout_rate, name="decoder_dropout_" + num)(expanding_conv)
        return dropout

    def build(self, **kwargs):
        """
        Structure of the unet
        :param kwargs: []
        :return: unet model with or without attention
        """
        # Inputs
        inputs = Input(shape=self.image_shape, batch_size=self.batch_size, name='inputs')

        # Contracting part - Encoder
        # First contracting convolutional block
        contracting_conv1, dropout1 = self.encoder_block(inputs, "1", filters=64)

        # Second contracting convolutional block
        contracting_conv2, dropout2 = self.encoder_block(dropout1, "2", filters=128)

        # Third contracting convolutional block
        contracting_conv3, dropout3 = self.encoder_block(dropout2, "3", filters=256)

        # Fourth contracting convolutional block
        contracting_conv4, dropout4 = self.encoder_block(dropout3, "4", filters=512)

        # Bottleneck - intermediate part
        bottleneck_conv = self.convolutional_block(dropout4, "bottleneck_conv", filters=1024)

        # Expanding part - Decoder
        # Fourth expanding deconvolution
        decoder_block4 = self.decoder_block(bottleneck_conv, contracting_conv4, 512, "4")

        # Third expanding deconvolution
        decoder_block3 = self.decoder_block(decoder_block4, contracting_conv3, 256, "3")

        # Second expanding deconvolution
        decoder_block2 = self.decoder_block(decoder_block3, contracting_conv2, 128, "2")

        # First expanding deconvolution
        decoder_block1 = self.decoder_block(decoder_block2, contracting_conv1, 64, "1")

        outputs = Conv2D(1, 1, activation='sigmoid', padding='same', name='output')(decoder_block1)

        model = Model(inputs=inputs, outputs=outputs, name="BayesianUNet")
        return model
