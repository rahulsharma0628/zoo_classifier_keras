import numpy as np
import os
import cv2
import argparse
import keras
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras import regularizers, optimizers

def data_preparation(dataset_path, label):

    cats = os.listdir(dataset_path + "/cats/")
    for cat in cats:
        img = cv2.imread(dataset_path + "/cats/" + cat)
        resized_image = np.resize(img, (200, 200, 3))
        data.append(np.array(resized_image))
        label.append(0)

    dogs = os.listdir(dataset_path + "/dogs/")
    for dog in dogs:
        img = cv2.imread(dataset_path + "/dogs/" + dog)
        resized_image = np.resize(img, (200, 200, 3))
        data.append(np.array(resized_image))
        label.append(1)

    pandas = os.listdir(dataset_path + "/panda/")
    for panda in pandas:
        img = cv2.imread(dataset_path + "/panda/" + panda)
        resized_image = np.resize(img, (200, 200, 3))
        data.append(np.array(resized_image))
        label.append(2)
    return data, label

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def plot_loss_accuracy(history,plot_path):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"], 'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"], 'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('Cross Entropy Loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["accuracy"], 'r-x', label="Train Accuracy")
    ax.plot(history.history["val_accuracy"], 'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('Accuracy')
    ax.grid(True)
    fig.savefig(plot_path)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="Input Path to Dataset")
parser.add_argument("-m", "--model", required=True, help="Path to output trained model")
parser.add_argument("-p", "--plot", required=True, help="path to output accuracy/loss plot")
args = vars(parser.parse_args())

data = []
labels = []
image_path = args['dataset']
model_path = args['model']
plot_path = args['plot']

data_params = data_preparation(image_path, labels)

animals = np.array(data_params[0])
labels = np.array(data_params[1])

X_train, X_test, y_train, y_test = train_test_split(animals, labels, test_size=0.2)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# One-hot encoding
y_train = keras.utils.to_categorical(y_train, 3)
y_test = keras.utils.to_categorical(y_test, 3)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
valid_datagen = ImageDataGenerator(
    rescale=1 / 255,
)
batch_size = 64

train_generator = train_datagen.flow_from_directory(
    image_path,
    target_size=(200, 200),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
)
validation_generator = train_datagen.flow_from_directory(
    image_path,
    target_size=(200, 200),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Model Generation using Sequential
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, activation="relu", input_shape=(200, 200, 3)))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, activation="relu"))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, activation="relu"))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.2))
model.add(Dense(3, activation="softmax"))
model.summary()

adam = optimizers.adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', precision_m, recall_m])

# history = model.fit(X_train, y_train, batch_size=128, epochs=50, verbose=1, validation_data=(X_test, y_test))
history = model.fit_generator(train_generator,
                              steps_per_epoch=40,
                              epochs=50,
                              validation_data=validation_generator,
                              validation_steps=40,
                              workers=5
                              )
model.save(model_path)

# model evaluation
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)

plot_loss_accuracy(history, plot_path)
plot_model(model)