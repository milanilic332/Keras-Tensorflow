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
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint


def main():
    # Size of images.
    img_width = 140
    img_height = 209

    # Directories of train and test sets.
    train_data_dir = 'data/train'
    test_data_dir = 'data/test'

    # ImageDataGenerator for train images (using data augmentation)
    datagen_train = ImageDataGenerator(rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       rescale=1./255,
                                       horizontal_flip=True,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       fill_mode='nearest')

    train_generator = datagen_train.flow_from_directory(directory=train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        classes=['male', 'female'],
                                                        class_mode='binary',
                                                        batch_size=4)

    # ImageDataGenerator for test images
    datagen_test = ImageDataGenerator(rescale=1./255)

    test_generator = datagen_test.flow_from_directory(directory=test_data_dir,
                                                      target_size=(img_width, img_height),
                                                      classes=['male', 'female'],
                                                      class_mode='binary',
                                                      batch_size=4)

    # Model architecture
    model = Sequential()

    # Conv1 layer
    model.add(Conv2D(128, (6, 9), strides=3, input_shape=(img_width, img_height, 3), activation='relu'))
    model.add(MaxPooling2D((3, 3)))

    # Conv2 layer
    model.add(Conv2D(256, (3, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D((3, 3)))

    # Conv3 layer
    model.add(Conv2D(512, (2, 3), activation='relu', padding='same'))

    # Conv4 layer
    model.add(Conv2D(256, (2, 3), activation='relu', padding='same'))

    # Conv5 layer
    model.add(Conv2D(128, (2, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    # FC1 layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.3))

    # FC2 layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    # FC3 layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(0.0001, decay=1e-8)

    # Compiling a model.
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Using TensorBoard callback for data visualization
    tbCallback = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)

    # Using ModelCheckpoint for saving the best model overall
    modelchkp = ModelCheckpoint('models/saved_model.h5', save_best_only=True)

    # Training a model.
    model.fit_generator(generator=train_generator, steps_per_epoch=13643//4, epochs=100,
                        validation_data=test_generator, validation_steps=6002//4,
                        callbacks=[tbCallback, modelchkp])


if __name__ == '__main__':
    main()
