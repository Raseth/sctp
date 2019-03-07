import numpy as np
import pandas as pd
import tensorflow as tf

from autoencoder import SymmetricAutoencoder, NumpyDataset

df = pd.read_csv('./data/train.csv', delimiter=',')

# shuffle and split dataset
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

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_dev = tf.keras.utils.normalize(x_dev, axis=1)

# AUTOENCODER
ae_input_size = 200
ae_train_dataset = NumpyDataset(np_array=x_train, batch_size=180000)
ae_test_dataset = NumpyDataset(np_array=x_test, batch_size=10000)
ae_dev_dataset = NumpyDataset(np_array=x_dev, batch_size=10000)

ae_model = SymmetricAutoencoder([ae_input_size, 100, 100, 50], noise_stddev=0.01)
opt = tf.keras.optimizers.Nadam(lr=0.0001)
ae_model.compile(optimizer=opt, loss='mse')
ae_model.summary()

ae_model.fit(ae_train_dataset._dataset, epochs=5, steps_per_epoch=1,
             validation_data=ae_test_dataset._dataset, validation_steps=1)

# ae_model.evaluate(ae_dev_dataset._dataset)

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
# x_dev = tf.keras.utils.normalize(x_dev, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)


val_loss_dev, val_acc_dev = model.evaluate(x_dev, y_dev)
print(val_loss_dev)
print(val_acc_dev)

predictions = model.predict(x_test)

a = 0

