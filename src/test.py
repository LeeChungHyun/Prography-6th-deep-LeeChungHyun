import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils import np_utils
from train import vgg_16
tf.enable_eager_execution()


def preprocess(test_images):
    mnist = tf.keras.datasets.mnist
    #학습에 사용될 부분과 테스트에 사용될 부분
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    #class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    #mnist_데이터는 0-255이므로 데이터를 0-1사이의 값을 만들기 위해 255로 나눈다.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #정규화 과정
    train_labels = np_utils.to_categorical(train_labels, 10, dtype='float32')
    test_labels = np_utils.to_categorical(test_labels, 10, dtype='float32')

    # Grayscale인 데이터를 RGB로 변환한다.
    train_images = np.stack(
        (train_images, train_images, train_images), axis=-1)
    test_images = np.stack((test_images, test_images, test_images,), axis=-1)
    return test_images


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    model = vgg_16()
    model.load_weights('./weights/prography')
    
    #모델을 평가해보자.
    test_images = preprocess(test_images)
    prediction = model.predict(test_images)
    pred_onehot = [tf.math.argmax(prediction[i]).numpy() for i in range(len(test_images))]
    acc = sum([1 for i in range(len(test_images)) if pred_onehot[i] == test_labels[i]]) / len(test_labels)
    print(f'Acc is: {acc*100}%')
