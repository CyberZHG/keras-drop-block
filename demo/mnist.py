import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import fashion_mnist
from keras_drop_block import DropBlock2D


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.expand_dims(x_train.astype(K.floatx()) / 255, axis=-1)
x_test = np.expand_dims(x_test.astype(K.floatx()) / 255, axis=-1)

y_train, y_test = np.expand_dims(y_train, axis=-1), np.expand_dims(y_test, axis=-1)

train_num = round(x_train.shape[0] * 0.9)
x_train, x_valid = x_train[:train_num, ...], x_train[train_num:, ...]
y_train, y_valid = y_train[:train_num, ...], y_train[train_num:, ...]


def get_dropout_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dropout(input_shape=(28, 28, 1), rate=0.3, name='Input-Dropout'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='Conv-1'))
    model.add(keras.layers.MaxPool2D(pool_size=2, name='Pool-1'))
    model.add(keras.layers.Dropout(rate=0.2, name='Dropout-1'))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv-2'))
    model.add(keras.layers.MaxPool2D(pool_size=2, name='Pool-2'))
    model.add(keras.layers.Dropout(rate=0.2, name='Dropout-2'))
    model.add(keras.layers.Flatten(name='Flatten'))
    model.add(keras.layers.Dense(units=256, activation='relu', name='Dense'))
    model.add(keras.layers.Dropout(rate=0.2, name='Dense-Dropout'))
    model.add(keras.layers.Dense(units=10, activation='softmax', name='Softmax'))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


dropout_model = get_dropout_model()
dropout_model.summary()
dropout_model.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    validation_data=(x_valid, y_valid),
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_acc', patience=2)]
)
dropout_score = dropout_model.evaluate(x_test, y_test)
print('Score of dropout:\t%.4f' % dropout_score[1])


def get_drop_block_model():
    model = keras.models.Sequential()
    model.add(DropBlock2D(input_shape=(28, 28, 1), block_size=7, keep_prob=0.8, name='Input-Dropout'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='Conv-1'))
    model.add(keras.layers.MaxPool2D(pool_size=2, name='Pool-1'))
    model.add(DropBlock2D(block_size=5, keep_prob=0.8, name='Dropout-1'))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv-2'))
    model.add(keras.layers.MaxPool2D(pool_size=2, name='Pool-2'))
    model.add(DropBlock2D(block_size=3, keep_prob=0.8, name='Dropout-2'))
    model.add(keras.layers.Flatten(name='Flatten'))
    model.add(keras.layers.Dense(units=256, activation='relu', name='Dense'))
    model.add(keras.layers.Dropout(rate=0.2, name='Dense-Dropout'))
    model.add(keras.layers.Dense(units=10, activation='softmax', name='Softmax'))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


drop_block_model = get_drop_block_model()
drop_block_model.summary()
drop_block_model.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    validation_data=(x_valid, y_valid),
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_acc', patience=2)]
)
drop_block_score = drop_block_model.evaluate(x_test, y_test)
print('Score of dropout:\t%.4f' % dropout_score[1])
print('Score of DropBlock:\t%.4f' % drop_block_score[1])
