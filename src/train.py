import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from tensorflow.keras import Model, layers
from keras.layers import Dense, Flatten, Input

#Data_Preprocessing
def preprocess(train_images):
    mnist = tf.keras.datasets.mnist
    #학습에 사용될 부분과 테스트에 사용될 부분
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    #mnist_데이터는 0-255이므로 데이터를 0-1사이의 값을 만들기 위해 255로 나눈다.
    train_images = train_images.reshape(60000, 784).astype('float32') / 255.0
    test_images = test_images.reshape(10000, 784).astype('float32') / 255.0
    #정규화 과정
     
    train_labels = np_utils.to_categorical(train_labels, 10, dtype='float32')
    test_labels = np_utils.to_categorical(test_labels, 10, dtype='float32')
    
    # Grayscale인 데이터를 RGB로 변환한다.
    train_images = np.stack((train_images) * 3, axis=-1)
    test_images = np.stack((test_images) * 3, axis=-1)
    return train_images
    #, train_labels, test_images, test_labels
    
#Model Initialization
class vgg_16(Model):
    def __init__(self):  # __init__()는 tf.keras.layers에서 환경변수 저장
        super(vgg_16, self).__init__()
        #_input = Input((224, 224, 3)) #input은 224x224x3의 이미지로 입력받는다.

        self.Conv1_1 = layers.Conv2D(64, 3, activation='relu', padding="same")
        self.Conv1_2 = layers.Conv2D(64, 3, activation='relu', padding="same")
        self.max_pool_1 = layers.MaxPooling2D(strides=(2, 2), padding="same")
        #VGG의 커널 사이즈는 3으로 고정한다.
        #stride를 2로 적용하므로 112x112x64의 activation map 사이즈 발생

        shortcut = self.max_pool_1
        shortcut = layers.Conv2D(512, 3, activation='relu', padding="same")
        shortcut = layers.MaxPooling2D(strides=(16, 16), padding="same")

        self.Conv2_1 = layers.Conv2D(128, 3, activation='relu', padding="same")
        self.Conv2_2 = layers.Conv2D(128, 3, activation='relu', padding="same")
        self.max_pool_2 = layers.MaxPooling2D(strides=(2, 2), padding="same")
        #stride를 2로 적용하므로 56x56x128의 activation map 사이즈 발생
        self.Conv3_1 = layers.Conv2D(256, 3, activation='relu', padding="same")
        self.Conv3_2 = layers.Conv2D(256, 3, activation='relu', padding="same")
        self.Conv3_3 = layers.Conv2D(256, 3, activation='relu', padding="same")
        self.max_pool_3 = layers.MaxPooling2D(strides=(2, 2), padding="same")
        #stride를 2로 적용하므로 28x28x256의 activation map 사이즈 발생
        self.Conv4_1 = layers.Conv2D(512, 3, activation='relu', padding="same")
        self.Conv4_2 = layers.Conv2D(512, 3, activation='relu', padding="same")
        self.Conv4_3 = layers.Conv2D(512, 3, activation='relu', padding="same")
        self.max_pool_4 = layers.MaxPooling2D(strides=(2, 2), padding="same")
        #stride를 2로 적용하므로 14x14x512의 activation map 사이즈 발생
        self.Conv5_1 = layers.Conv2D(512, 3, activation='relu', padding="same")
        self.Conv5_2 = layers.Conv2D(512, 3, activation='relu', padding="same")
        self.Conv5_3 = layers.Conv2D(512, 3, activation='relu', padding="same")
        self.max_pool_5 = layers.MaxPooling2D(strides=(2, 2), padding="same")
        #stride를 2로 적용하므로 7x7x512의 activation map 사이즈 발생

        self.flat = layers.Flatten()
        self.shortcut_flat = layers.Flatten()

        #fc1부터는 전체 activation map을 1차원 벡터로 펼쳐준다(flatten)
        #총 25088개의 뉴런 생성+fc1층의 4096개의 뉴런과 fc된다.
        self.fc1 = layers.Dense(4096)
        self.fc2 = layers.Dense(4096)
        self.fc3 = layers.Dense(10, activation='softmax')
        #mnist 데이터는 0-9, 10개의 클래스로 구성하므로 10개의 뉴런으로 구성된다.
        #shortcut connection은 layer의 입력을 layer의 출력에 바로 연결시키는 기법이다.
        #그니까 layer통해 나온 결과와 그 전의 결과를 더한다.

    def shortcut_connection(self, inputs):
        x = self.Conv1_1(inputs)
        x = self.Conv1_2(x)
        x = self.max_pool_1(x)
        shortcut = x

        x = self.Conv2_1(x)
        x = self.Conv2_2(x)
        x = self.max_pool_2(x)

        x = self.Conv3_1(x)
        x = self.Conv3_2(x)
        x = self.Conv3_3(x)
        x = self.max_pool_3(x)

        x = self.Conv4_1(x)
        x = self.Conv4_2(x)
        x = self.Conv4_3(x)
        x = self.max_pool_4(x)

        x = self.Conv5_1(x)
        x = self.Conv5_2(x)
        x = self.Conv5_3(x)
        x = self.max_pool_5(x)

        x = self.flat(x)
        shortcut = self.shortcut_flat(shortcut)
        x = tf.concat([x, shortcut], axis=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

#Model_Training
if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    model = vgg_16()
    train_images = preprocess(train_images)
   
    #모델의 학습과정을 설정한다.
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    epochs = 5
    batch_size = 64

    #모델을 학습시킨다.
    history = model.fit(train_images, train_labels, epochs=epochs)
    
    #학습된 weight 저장한다.
    model.save_weights('./checkpoints', save_format='tf')

    #학습과정을  살펴본다.
    print('## Training Accuracy ##')
    print(history.history['Acc'])
  
    #모델을 평가한다.
    loss_and_metrics = model.evaluate(test_images, test_labels, batch_size=batch_size)
    print('## Evaluation loss and_metrics ##')
    print(loss_and_metrics)


 
