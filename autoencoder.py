import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GaussianNoise, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model


class SymmetricAutoencoder:

    def __init__(self, dims, act='relu', init='glorot_uniform', noise_stddev=0.0):
        n_stacks = len(dims) - 1
        # input
        x = Input(shape=(dims[0],), name='input')
        h = GaussianNoise(noise_stddev)(x)

        # internal layers in encoder
        for i in range(n_stacks - 1):
            h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)
            # h = BatchNormalization(momentum=0.66)(h)
            # h = Dropout(0.3)(h)

        # hidden layer
        h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(
            h)  # hidden bottleneck layer, features are extracted from here

        y = h
        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)
            # y = BatchNormalization(momentum=0.66)(y)
            # y = Dropout(0.3)(y)

        # output
        y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

        self.model = Model(inputs=x, outputs=y, name='AE')
        self.encoder = Model(inputs=x, outputs=h, name='encoder')

    def load_weights(self, weights):
        self.model.load_weights(weights, by_name=True)

    def load_encoder(self, model):
        self.encoder = load_model(model)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        y = self.model.predict(x, verbose=0, steps=1)
        return y

    def compile(self, optimizer='sgd', loss='mse'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def summary(self):
        self.model.summary()

    def evaluate(self, x):
        val_loss, val_acc = self.model.evaluate(x)
        print(val_loss)
        print(val_acc)

    def fit(self, dataset, epochs=10, steps_per_epoch=30, validation_data=None, validation_steps=None,
            save_dir='results/', file_name='ae_weights.h5', log_name='Autoencoder'):

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=(save_dir + "{}".format(log_name)))
        checkpoint = ModelCheckpoint(save_dir + "ae_weights.{epoch:02d}-{loss:.5f}.h5", monitor='loss',
                                     save_weights_only=True, period=10)

        self.model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=validation_data,
                       validation_steps=validation_steps, callbacks=[checkpoint, tensorboard])

        self.model.save_weights(save_dir + file_name)
        print('Autoencoder weights are saved to %s/%s', save_dir, file_name)


class NumpyDataset:
    def __init__(self, np_array, batch_size=128):
        dataset = tf.data.Dataset.from_tensor_slices((np_array, np_array))
        dataset = dataset.repeat()
        self.x = np_array
        self.y = np_array
        self.number_of_examples = len(np_array)
        self._dataset = dataset.batch(batch_size)
        self.input_size = dataset.output_shapes[0][-1]
        self.batch_size = batch_size

    def get_steps_per_epoch(self):
        return self.number_of_examples // self.batch_size
