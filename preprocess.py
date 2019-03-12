import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GaussianNoise, BatchNormalization, Dropout


df = pd.read_csv('./data/train.csv', delimiter=',')
df_test = pd.read_csv('./data/test.csv', delimiter=',')


# shuffle and split train dataset
df = df.sample(frac=1).reset_index(drop=True)
train = df.sample(frac=0.9, random_state=200)
test = df.drop(train.index)
dev = test.sample(frac=0.5, random_state=200)
test = test.drop(dev.index)
print(len(train), len(test), len(dev))

# copying rows with target 1, so both classes will have same amount of data rows
print((train[train['target'] == 0].shape[0] - train[train['target'] == 1].shape[0]) /
      train[train['target'] == 1].shape[0])
train.reset_index(inplace=True)
train_one = pd.concat([train[train['target'] == 1]]*8, ignore_index=True)
train = train.append([train_one], ignore_index=True)
train.drop('index', axis=1, inplace=True)
print(train['target'].value_counts())


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
x_output = tf.keras.utils.normalize(x_output, axis=1)


x = Input(shape=(200,), name='input')  # jak z VAE, to to zakomentowac
h = GaussianNoise(stddev=0.01)(x)
h = Dense(256, activation=tf.nn.relu, kernel_initializer='glorot_uniform', name='layer_1')(h)
h = BatchNormalization(momentum=0.66)(h)
h = Dropout(0.3)(h)
h = Dense(128, activation=tf.nn.relu, kernel_initializer='glorot_uniform', name='layer_2')(h)
h = Dense(2, activation=tf.nn.softmax, kernel_initializer='glorot_uniform', name='output')(h)

model = tf.keras.models.Model(inputs=x, outputs=h)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=50)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)


val_loss_dev, val_acc_dev = model.evaluate(x_dev, y_dev)
print(val_loss_dev)
print(val_acc_dev)

predictions = model.predict(x_output)
predictions_classes = np.argmax(predictions, axis=1)

df_output = pd.read_csv('./data/sample_submission.csv', delimiter=',')
df_output['target'] = predictions_classes
df_output.to_csv('./data/submission.csv', sep=',', index=False)
a = 0
