import numpy as np
import pandas as pd
import tensorflow as tf


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





a = 0

