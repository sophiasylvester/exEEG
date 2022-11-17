import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time

import os
import random
from tensorflow.python.eager import context
from tensorflow.python.framework import ops

# seeds
seed_value = 2
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)
np.random.RandomState(seed_value)
np.random.seed(seed_value)
context.set_global_seed(seed_value)
ops.get_default_graph().seed = seed_value
# installation of tensorflow-determinism required
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def split_data(X, y, name):
    size = 0.1
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=size, random_state=seed_value)
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=size, random_state=seed_value)
    train_name = 'data/X_train_' + name + '.npy'
    test_name = 'data/X_test_' + name + '.npy'
    with open(train_name, 'wb') as f:
        np.save(f, X_train)
    with open(test_name, 'wb') as g:
        np.save(g, X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_model(gamma):
    inputs = tf.keras.layers.Input(shape=(768, 128), name='input')

    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=(21), strides=1, activation='relu',
                                         kernel_initializer='he_normal',
                                         kernel_regularizer=tf.keras.regularizers.L2(l=0.01))(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=(2), strides=2)(x)
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=(21), strides=1, activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.L2(l=0.01))(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=(2), strides=2)(x)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=(21), strides=1, activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.L2(l=0.01))(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=(2), strides=2)(x)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=(21), strides=1, activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.L2(l=0.01))(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=(2), strides=2)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=(21), strides=1, activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.L2(l=0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    x_1 = tf.keras.layers.Dense(128, activation='relu')(x)
    x_1 = tf.keras.layers.Dense(64, activation='relu')(x_1)
    x_1 = tf.keras.layers.Dense(26, activation='softmax', name='out_1')(x_1)

    x_2 = tf.keras.layers.Dense(128, activation='relu')(x)
    x_2 = tf.keras.layers.Dense(64, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(32, activation='relu')(x_2)
    x_2 = tf.keras.layers.Dense(2, activation='sigmoid', name='out_2')(x_2)

    model = tf.keras.Model(inputs=inputs, outputs=[x_1, x_2])

    return model


def compile_multitask_model(model, gamma):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,  # 1e-2 = 0.01 # 3 works with 0.79 acc
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.99,
                                         epsilon=1e-9)
    model.compile(optimizer=optimizer,
                  loss={'out_1': 'sparse_categorical_crossentropy',
                        'out_2': 'sparse_categorical_crossentropy'},
                  loss_weights={'out_1': gamma,
                                'out_2': 1 - gamma},
                  metrics=['accuracy'])

    return model


def fit_batch(gamma_values, X_train, y_train, X_test, y_test):
    history = list()
    trained_models = list()

    print('Starting training on batch of models for gamma values ', gamma_values, '\n\n')

    for gamma in gamma_values:
        print('\n-----------------------------Training model for gamma ', gamma, "-------------------------\n")
        model = create_model(gamma)
        model = compile_multitask_model(model, gamma)
        estop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        checkpoint_filepath = 'cnnmlt_checkpoints/cnnmlt_check'
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_out_2_accuracy',
            mode='max',
            save_best_only=True)
        model_history = model.fit({'input': X_train},
                                  {'out_1': y_train[:,1], 'out_2': y_train[:,0]},
                                  epochs=100, batch_size=64, verbose=0, validation_data=(X_val, [y_val[:,1], y_val[:,0]]),
                                  callbacks=[estop_callback, checkpoint_callback])
        history.append(model_history)
        trained_models.append(model)
        print("Test results:")
        model.load_weights(checkpoint_filepath)
        model.evaluate(X_test, (y_test[:,1],y_test[:,0]), batch_size=64, verbose=2)

    return history, trained_models


if __name__ == '__main__':

    t_start = time()
    print("\n\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)

    t0 = time()
    X = np.load("data/X_face.npy")
    y_bin = np.load("data/y_face.npy")
    y_vp = np.repeat(np.arange(26), 120)
    y = np.vstack((y_bin, y_vp))
    y = np.swapaxes(y,0,1)
    print("X shape: ", X.shape, "y shape: ", y.shape, "n face target classes: ", np.unique(y[:,0]), "\n", flush=True)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, 'face')

    # model = create_model(0.5)
    gammas = [0, 0.3, 0.5, 0.7, 1.0]
    histories, trained_models = fit_batch(gammas, X_train, y_train, X_test, y_test)

    print("\nTime to train model: ", (time() - t0) / 60, "mins")