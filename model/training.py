import tensorflow as tf
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from keras.optimizers.legacy import Adam
from tensorflow.data import AUTOTUNE

from model.bayesian_unet import BayesianUnet
from model.losses import combined_loss, dice_coefficient
from db_utils import get_all_train_data_for_training
from image_utils import read_frame_from_dicom_for_training, create_binary_mask_for_training


class Training:

    def __init__(self, mode, num_epochs, start_lr, dropout_rate, db_config):
        """
        Initialization of training instance
        :param mode: mode of usage train| train_scratch
        :param num_epochs: number of epochs used for training
        :param start_lr: learning rate to start with (default - 0.0001)
        :param dropout_rate: dropout rate of CustomDropout Layers (default - 0.1)
        :param db_config: Configuration file used for the connection to echo db
        """

        # Model configs
        self.dropout_rate = dropout_rate
        self.mode = mode
        self.loss = combined_loss
        self.optimizer = Adam(learning_rate=start_lr)
        self.epochs = num_epochs
        self.metrics = [dice_coefficient, 'binary_crossentropy']

        # Training conditions
        self.strategy = tf.distribute.MirroredStrategy()
        self.batch_size = len(tf.config.list_physical_devices('GPU'))

        # Get annotated traces from the database
        self.manually_annotated_frames = get_all_train_data_for_training(db_config)
        self.train_set, self.val_set = train_test_split(self.manually_annotated_frames, test_size=0.2, random_state=12,
                                                        shuffle=True)
        # Calculate stats
        self.steps_per_epoch = len(self.train_set) // self.batch_size
        self.validation_steps = len(self.val_set) // self.batch_size

        # Build the training data sets and masks
        self.tf_train_set = self.create_dataset(self.train_set)
        # self.tf_train_mask_set = tf.data.Dataset.from_tensor_slices(self.train_masks)
        self.tf_train_mask_set = self.create_masks(self.train_set)

        # Merge frames and masks
        self.tf_train_set = (
            tf.data.Dataset.zip((self.tf_train_set, self.tf_train_mask_set)).batch(self.batch_size).prefetch(AUTOTUNE))

        # Build the validation data sets and masks
        self.tf_val_set = self.create_dataset(self.val_set)
        self.tf_val_mask_set = self.create_masks(self.val_set)

        # Merge frames and masks
        self.tf_val_set = tf.data.Dataset.zip((self.tf_val_set, self.tf_val_mask_set)).batch(self.batch_size).prefetch(
            AUTOTUNE)

        # Build model
        self.model = None
        if self.batch_size <= 1:
            self.build_and_compile_model()
        else:
            with self.strategy.scope():
                self.distributed_train_set = self.strategy.experimental_distribute_dataset(self.tf_train_set)
                self.distributed_test_set = self.strategy.experimental_distribute_dataset(self.tf_val_set)
                self.build_and_compile_model()

        # Callbacks
        self.training_callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=30),
            LearningRateScheduler(self.scheduler, verbose=1),
        ]

    def build_and_compile_model(self):
        """
        Build and compile the model
        """
        self.model = BayesianUnet(mode=self.mode, dropout_rate=self.dropout_rate, batch_size=self.batch_size)
        self.model = self.model.build()
        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        print("Model compiled")

    def train(self):
        """
        Implements the model instantiation and the training routine
        """
        if self.batch_size <= 1:
            self.model.fit(self.tf_train_set, epochs=self.epochs, validation_data=self.tf_val_set,
                           callbacks=self.training_callbacks)
        else:
            with self.strategy.scope():
                self.model.fit(self.distributed_train_set, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch,
                               validation_data=self.distributed_test_set, validation_steps=self.validation_steps,
                               callbacks=self.training_callbacks)

    @staticmethod
    def create_dataset(data_set, padding='constant'):
        """
        This method creates a Tensorflow data input pipeline.
        :param padding: constant | edge
        :param data_set: pandas data frame
        :return: tensorflow data set
        """

        file_paths = data_set['root'] + data_set['path_op'] + data_set['path_from_node'] + "/" + data_set['file_name']

        tf_data_set = tf.data.Dataset.from_tensor_slices(
            (file_paths, data_set['fr_number'], data_set['x0'], data_set['y0'], data_set['x1'], data_set['y1']))
        tf_data_set = tf_data_set.map(
            lambda file_path, frame_number, x0, y0, x1, y1: tf.py_function(read_frame_from_dicom_for_training,
                                                                           (file_path, frame_number, x0, y0, x1, y1,
                                                                            padding), tf.int32),
            num_parallel_calls=AUTOTUNE
        )
        return tf_data_set

    @staticmethod
    def create_masks(data_set):
        tf_masks = tf.data.Dataset.from_tensor_slices(
            (data_set['tr_all_points_x'], data_set['tr_all_points_y'], data_set['x0'], data_set['y0'], data_set['x1'],
             data_set['y1']))
        tf_masks = tf_masks.map(
            lambda tr_all_points_x, tr_all_points_y, x0, y0, x1, y1: tf.py_function(create_binary_mask_for_training,
                                                                                    (tr_all_points_x, tr_all_points_y,
                                                                                     x0, y0, x1, y1), tf.int32),
            num_parallel_calls=AUTOTUNE)
        return tf_masks

    @staticmethod
    def scheduler(epoch, lr):
        """
        Learning rate scheduler
        :param epoch: current epoch
        :param lr: current learning rate
        :return: next learning rate to use for training
        """
        if epoch < 25:
            return lr
        if epoch < 80:
            return 0.000001
        else:
            return 0.0000001
