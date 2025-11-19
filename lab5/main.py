import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras import layers

# Загружаем тренировочные и тестовые данные
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

# Просмотр изображения
plt.figure()
plt.imshow(x_train[100])
plt.colorbar()
plt.grid(False)

# Нормализация данных
x_train = x_train / 255
x_test = x_test / 255

# Создание модели
model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3),input_shape=(32,32,3) ),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(128,(3,3), activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),

    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(100, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()


model.fit(x_train, y_train, epochs=50)


test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)