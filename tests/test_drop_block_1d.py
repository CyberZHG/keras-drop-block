import os
import random
import tempfile
import unittest

import numpy as np
from tensorflow import keras

from keras_drop_block import DropBlock1D


class TestDropBlock1D(unittest.TestCase):

    def test_training(self):
        input_layer = keras.layers.Input(shape=(100, 3))
        drop_block_layer = DropBlock1D(block_size=3, keep_prob=0.7)(input_layer)
        model = keras.models.Model(inputs=input_layer, outputs=drop_block_layer)
        model.compile(optimizer='adam', loss='mse', metrics={})
        model_path = os.path.join(tempfile.gettempdir(), 'keras_drop_block_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={'DropBlock1D': DropBlock1D},
        )
        model.summary()
        inputs = np.ones((1, 100, 3))
        outputs = model.predict(inputs)
        self.assertTrue(np.allclose(inputs, outputs))

        input_layer = keras.layers.Input(shape=(3, 100))
        drop_block_layer = DropBlock1D(block_size=3, keep_prob=0.7, data_format='channels_first')(input_layer)
        model = keras.models.Model(inputs=input_layer, outputs=drop_block_layer)
        model.compile(optimizer='adam', loss='mse', metrics={})
        model_path = os.path.join(tempfile.gettempdir(), 'keras_drop_block_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={'DropBlock1D': DropBlock1D},
        )
        model.summary()
        inputs = np.ones((1, 3, 100))
        outputs = model.predict(inputs)
        self.assertTrue(np.allclose(inputs, outputs))

    def test_mask_shape(self):
        input_layer = keras.layers.Input(shape=(100, 3))
        drop_block_layer = DropBlock1D(block_size=3, keep_prob=0.7)(input_layer, training=True)
        model = keras.models.Model(inputs=input_layer, outputs=drop_block_layer)
        model.compile(optimizer='adam', loss='mse', metrics={})
        model_path = os.path.join(tempfile.gettempdir(), 'keras_drop_block_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={'DropBlock1D': DropBlock1D},
        )
        model.summary()
        inputs = np.ones((1, 100, 3))
        outputs = model.predict(inputs)
        for i in range(3):
            print((outputs[0, :, i] > 0.0).astype(dtype='int32'))
        inputs = np.ones((1000, 100, 3))
        outputs = model.predict(inputs)
        keep_prob = 1.0 * np.sum(outputs > 0.0) / np.prod(np.shape(outputs))
        print(keep_prob)
        self.assertTrue(0.65 < keep_prob < 0.8, keep_prob)

        input_layer = keras.layers.Input(shape=(3, 100))
        drop_block_layer = DropBlock1D(block_size=3, keep_prob=0.7,
                                       data_format='channels_first')(input_layer, training=True)
        model = keras.models.Model(inputs=input_layer, outputs=drop_block_layer)
        model.compile(optimizer='adam', loss='mse', metrics={})
        model_path = os.path.join(tempfile.gettempdir(), 'keras_drop_block_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={'DropBlock1D': DropBlock1D},
        )
        model.summary()
        inputs = np.ones((1, 3, 100))
        outputs = model.predict(inputs)
        for i in range(3):
            print((outputs[0, i, :] > 0.0).astype(dtype='int32'))
        inputs = np.ones((1000, 3, 100))
        outputs = model.predict(inputs)
        keep_prob = 1.0 * np.sum(outputs > 0.0) / np.prod(np.shape(outputs))
        print(keep_prob)
        self.assertTrue(0.65 < keep_prob < 0.8, keep_prob)

    def test_sync_channels(self):
        input_layer = keras.layers.Input(shape=(100, 3))
        drop_block_layer = DropBlock1D(block_size=3, keep_prob=0.7, sync_channels=True)(input_layer, training=True)
        model = keras.models.Model(inputs=input_layer, outputs=drop_block_layer)
        model.compile(optimizer='adam', loss='mse', metrics={})
        model_path = os.path.join(tempfile.gettempdir(), 'keras_drop_block_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects={'DropBlock1D': DropBlock1D},
        )
        model.summary()
        inputs = np.ones((1, 100, 3))
        outputs = model.predict(inputs)
        for i in range(1, 3):
            self.assertTrue(np.allclose(outputs[0, :, 0], outputs[0, :, i]))
        inputs = np.ones((1000, 100, 3))
        outputs = model.predict(inputs)
        keep_prob = 1.0 * np.sum(outputs > 0.0) / np.prod(np.shape(outputs))
        print(keep_prob)
        self.assertTrue(0.65 < keep_prob < 0.8, keep_prob)
