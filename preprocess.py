import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, GaussianNoise, BatchNormalization, Dropout

from autoencoder import SymmetricAutoencoder, VariationalAutoencoder, NumpyDataset

df = pd.read_csv('./data/train.csv', delimiter=',')
df_test = pd.read_csv('./data/test.csv', delimiter=',')


# shuffle and split train dataset
df = df.sample(frac=1).reset_index(drop=True)
train = df.sample(frac=0.9, random_state=200)
test = df.drop(train.index)
dev = test.sample(frac=0.5, random_state=200)
test = test.drop(dev.index)
print(len(train), len(test), len(dev))


ID_code_train = train['ID_code'].values
y_train = train['target'].values
train.drop(['ID_code', 'target'], axis=1, inplace=True)
x_train = train.values

ID_code_test = test['ID_code'].values
y_test = test['target'].values
test.drop(['ID_code', 'target'], axis=1, inplace=True)
x_test = test.values

ID_code_dev = dev['ID_code'].values
y_dev = dev['target'].values
dev.drop(['ID_code', 'target'], axis=1, inplace=True)
x_dev = dev.values

ID_code_output = df_test['ID_code'].values
df_test['target'] = 0
y_output = df_test['target'].values
df_test.drop(['ID_code', 'target'], axis=1, inplace=True)
x_output = df_test.values


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_dev = tf.keras.utils.normalize(x_dev, axis=1)


x = Input(shape=(200,), name='input')  # jak z VAE, to to zakomentowac
h = Dense(256, activation=tf.nn.relu, kernel_initializer='glorot_uniform', name='layer_1')(x)
# h = BatchNormalization(momentum=0.66)(h)
# h = Dropout(0.3)(h)
h = Dense(128, activation=tf.nn.relu, kernel_initializer='glorot_uniform', name='layer_2')(h)
h = Dense(2, activation=tf.nn.softmax, kernel_initializer='glorot_uniform', name='output')(h)

model = tf.keras.models.Model(inputs=x, outputs=h)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # returns probability
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)


val_loss_dev, val_acc_dev = model.evaluate(x_dev, y_dev)
print(val_loss_dev)
print(val_acc_dev)

predictions = model.predict(x_test)

predictions_classes = np.argmax(predictions, axis=1)

a = 0
