from keras.preprocessing import image
import numpy as np


def prepare_image(path, target_size=(32, 32)):
    img = image.load_img(path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array /= 255
    return img_array


def print_prediction(predictions, class_labels):
    for prediction in predictions:
        answer_idx = int(np.argmax(prediction))
        answer_confidence = prediction[answer_idx]

        print("Class: " + class_labels[answer_idx] +
              ", Confidence: " + str(answer_confidence))
