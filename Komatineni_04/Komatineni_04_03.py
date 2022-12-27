# Komatineni , Harshitha
# 1001-968-082
# 2022_11_13
# Assignment-04-03

import pytest
import numpy as np
from cnn import CNN
import os

def test_train_and_evaluate():
    from tensorflow.keras.datasets import mnist
    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape and Normalize data
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_test = y_test.flatten().astype(np.int32)
    input_dim = X_train.shape[1]
    indices = list(range(X_train.shape[0]))
    # np.random.shuffle(indices)
    number_of_samples_to_use_for_training = 500
    number_of_samples_to_use_for_testing = 100
    X_train = X_train[indices[:number_of_samples_to_use_for_training]]
    y_train = y_train[indices[:number_of_samples_to_use_for_training]]
    X_test = X_test[indices[:number_of_samples_to_use_for_testing]]
    y_test = y_test[indices[:number_of_samples_to_use_for_testing]]
    my_cnn = CNN()
    my_cnn.add_input_layer(shape=input_dim, name="input0")
    my_cnn.append_dense_layer(num_nodes=10, activation='linear', name="layer1")
    w = my_cnn.get_weights_without_biases(layer_name="layer1")
    w_set = np.full_like(w, 2)
    my_cnn.set_weights_without_biases(w_set, layer_name="layer1")
    b=my_cnn.get_biases(layer_name="layer1")
    b_set= np.full_like(b, 2)
    b_set[0]=b_set[0]*2
    my_cnn.set_biases(b_set, layer_name="layer1")
    my_cnn.set_loss_function("SparseCategoricalCrossentropy")
    my_cnn.set_metric("accuracy")
    my_cnn.set_optimizer("SGD")
    actual = my_cnn.train(X_train,y_train,100,4)
    evaluate = my_cnn.evaluate(X_test,y_test)
    np.testing.assert_almost_equal(actual,np.array([2.3025, 2.3025, 2.3025, 2.3025]),decimal=4)
    np.testing.assert_almost_equal(evaluate,np.array([2.3025, 0.08]),decimal=4)



