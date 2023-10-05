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

img = image.load_img('meme_cat.png', target_size=(32, 32))
img_to_test = image.img_to_array(img) / 255

list_of_images = np.expand_dims(img_to_test, axis=0)

prediction = model.predict(list_of_images)
answer_idx = int(np.argmax(prediction[0]))
answer_confidence = prediction[0][answer_idx]

print("Class: " + class_labels[answer_idx] +
      ", Confidence: " + str(answer_confidence))
