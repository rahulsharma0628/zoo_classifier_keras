import numpy as np
import cv2
import argparse
import keras
from keras import backend as K

def image_to_array(img):
    img = cv2.imread(img)
    resized_image = np.resize(img, (200, 200, 3))
    return resized_image

def get_animal(label):
    if label == 0:
        return "Cat"
    if label == 1:
        return "Dog"
    if label == 2:
        return "Panda"

def predict_animal(file, model):
    print("Predicting the animal in the image ----------------")
    ar = image_to_array(file)
    ar = ar/255
    a = [ar]
    a = np.array(a)
    score = model.predict(a, verbose=1)
    # print(score)
    label_index = np.argmax(score)
    # print(label_index)
    acc = np.max(score)
    animal = get_animal(label_index)
    print("The image input is predicted as a " + animal + " with the accuracy =    " + str(acc))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Path to input image for prediction")
    parser.add_argument("-m", "--model", required=True, help="Path to input trained model")
    args = vars(parser.parse_args())

    model = keras.models.load_model(args['model'], custom_objects={'precision_m':precision_m, 'recall_m':recall_m})
    predict_animal(args['image'], model)