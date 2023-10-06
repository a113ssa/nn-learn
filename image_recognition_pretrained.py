import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16


model = vgg16.VGG16()

img = image.load_img('canyon.png', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = vgg16.preprocess_input(img)

predictions = model.predict(img)
predicted_classes = vgg16.decode_predictions(predictions, top=3)

for _, name, likelihood in predicted_classes[0]:
    print(f'{name} - Likelihood: {likelihood:.2f}')
