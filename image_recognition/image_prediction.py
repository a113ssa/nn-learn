import numpy as np
from keras.models import model_from_json
from support_methods import prepare_image, print_prediction
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

model = model_from_json(
    Path('image_recognition/saved_model/model_structure.json').read_text())

model.load_weights('image_recognition/saved_model/model_weights.h5')

img1 = prepare_image('images/meme_cat.png')
img2 = prepare_image('images/meme_roach.png')

list_of_images = [np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)]

predictions = model.predict(np.vstack(list_of_images))

print_prediction(predictions, class_labels)
