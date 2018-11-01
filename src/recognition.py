from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from matplotlib import pyplot as plt
from keras.utils import np_utils
import numpy as np
import cv2
import os

from keras import backend as K

K.set_image_dim_ordering('th')

# imagem_cursiva = "seis.png"

# %matplotlib inline

seed = 7
np.random.seed(seed)

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')

    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

    x_train = x_train / 255
    x_test = x_test / 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()

num_classes = y_test.shape[1]


# O model será exportado para este arquivo
# filename='mnistneuralnet.h5'

def model():
    model = Sequential()

    # model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(30, (3, 3), input_shape=(1, 28, 28), activation='relu'))

    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(15, (3, 3), input_shape=(1, 28, 28), activation='relu'))

    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(15, (1, 1), input_shape=(1, 28, 28), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(num_classes, activation='softmax', name='predict'))

    #loss='categorical_crossentropy' função de perda
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# model.summary()

def execute(imagem, model =model()):
    global x_train, y_train, x_test, y_test
    filename = 'mnistneuralnet.h5'
# Verifica se já existe um modelo treinado e exportado para um arquivo .h5.
# Um novo modelo será treinado, caso este arquivo não exista.
    if not os.path.exists('./{}'.format(filename)):
        #validation_data=(x_test, y_test): faz a validação após a classificação
        #epochs (literalmente épocas): mesma coisa de ciclos/interações
        #batch_size: tamanho
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=200)
        model.save_weights(filename)
    else:
        # carrega um modelo previamente treinado
        model.load_weights('./{}'.format(filename))

    # scores = model.evaluate(x_test, y_test, verbose=0)
    # print("\nacc: %.2f%%" % (scores[1] * 100))

    img_pred = cv2.imread(imagem, 0)
    # plt.imshow(img_pred, cmap='gray')


    if img_pred.shape != [28,28]:
        img2 = cv2.resize(img_pred, (28, 28))
        img_pred = img2.reshape(28, 28, -1)
    else:
        img_pred = img_pred.reshape(28, 28, -1)

    img_pred = img_pred.reshape(1, 1, 28, 28).astype('float32')

    img_pred = img_pred/255.0

    pred = model.predict_classes(img_pred)
    pred_proba = model.predict_proba(img_pred)
    pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)
    print(pred[0], " com confiança de ", pred_proba)
    return "{} com confiança de {}".format(pred[0], pred_proba)
