from keras.datasets import mnist
import numpy as np
np.random.seed(50)
import matplotlib.pyplot as plt


def extract_100_images():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.reshape(
        x_train.shape[0], -1)/255.0, x_test.reshape(x_test.shape[0], -1)/255.0
    temp = []
    temp_target = []
    for i in range(10):
        temp.extend(x_train[np.where(y_train == i)[
                    0][np.random.permutation(100)]])
        temp_target.extend([i]*100)
    train = np.hstack((np.vstack(temp), np.vstack(temp_target)))
    test = np.hstack((x_test[:50], y_test[:50, None]))
    print(f"Train: {train.shape} Test: {test.shape}")
    np.random.shuffle(train)
    np.random.shuffle(test)
    return train, test
