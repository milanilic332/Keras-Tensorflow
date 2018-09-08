"""
    Making CNN with dataset of images.
    Predicting gender.
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


def main():
    # Size of images.
    img_width = 140
    img_height = 209

    # Directories of train and test sets.
    train_data_dir = 'data/train'
    test_data_dir = 'data/test'

    # ImageDataGenerator for classification (train and test).
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(directory=train_data_dir,
                                                  target_size=(img_width, img_height),
                                                  classes=['male', 'female'],
                                                  class_mode='binary',
                                                  batch_size=16)

    test_generator = datagen.flow_from_directory(directory=test_data_dir,
                                                 target_size=(img_width, img_height),
                                                 classes=['male', 'female'],
                                                 class_mode='binary',
                                                 batch_size=32)

    # Model architecture
    model = Sequential()

    # Conv1 layer
    model.add(Conv2D(16, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Conv2 layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv3 layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv4 layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # FC1 layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # FC2 layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compiling a model.
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Training a model.
    model.fit_generator(generator=train_generator, steps_per_epoch=6956//16, epochs=10,
                        validation_data=test_generator, validation_steps=3000//32)

    # Saving weights of a model
    model.save_weights('models/final_CNN.h5')


if __name__ == '__main__':
main()
