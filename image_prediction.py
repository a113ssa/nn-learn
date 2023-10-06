from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

from pathlib import Path

class_labels = [
    'Plane',
    'Car',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Boat',
    'Truck',
]

model = model_from_json(Path('model_structure.json').read_text())

model.load_weights('model_weights.h5')


def prepare_image(path):
    img = image.load_img(path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array /= 255
    return img_array


img1 = prepare_image('meme_cat.png')
img2 = prepare_image('meme_roach.png')

list_of_images = [np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)]

predictions = model.predict(np.vstack(list_of_images))

for prediction in predictions:
    answer_idx = int(np.argmax(prediction))
    answer_confidence = prediction[answer_idx]

    print("Class: " + class_labels[answer_idx] +
          ", Confidence: " + str(answer_confidence))
