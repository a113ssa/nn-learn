import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# convert 8 bit values to floats between 0 and 1
def normalize_data(data):
    return data.astype('float32') / 255


x_train = normalize_data(x_train)
x_test = normalize_data(x_test)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# create model and add layers
model = Sequential(
    [
        Conv2D(32, (3, 3), padding='same',
               activation='relu', input_shape=(32, 32, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax'),
    ]
)

# compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

# train model
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=30,
    validation_data=(x_test, y_test),
    shuffle=True,
)

# save model structure to json
Path('model_structure.json').write_text(model.to_json())

# save model weights to hdf5
model.save_weights('model_weights.h5')

model.summary()
