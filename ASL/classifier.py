"""
    Making CNN for classification of images from
    https://www.kaggle.com/grassknoted/asl-alphabet.
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
    img_width = 200
    img_height = 200

    # Classes of images
    classes = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I'
               'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R',
               'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # Directories of train and test sets.
    train_data_dir = 'data/asl_alphabet_train'
    test_data_dir = 'data/asl_alphabet_test'

    # ImageDataGenerator for train images (using data augmentation)
    datagen_train = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       fill_mode='nearest')

    train_generator = datagen_train.flow_from_directory(directory=train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        classes=classes,
                                                        class_mode='categorical',
                                                        batch_size=16)

    # ImageDataGenerator for test images
    datagen_test = ImageDataGenerator(rescale=1./255)

    test_generator = datagen_test.flow_from_directory(directory=test_data_dir,
                                                      target_size=(img_width, img_height),
                                                      classes=classes,
                                                      class_mode='categorical',
                                                      batch_size=16)

    # Model architecture
    model = Sequential()

    # Conv1 layer
    model.add(Conv2D(128, (9, 9), strides=3, input_shape=(img_width, img_height, 3), activation='relu'))
    model.add(MaxPooling2D((3, 3)))

    # Conv2 layer
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D((3, 3)))

    # Conv3 layer
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

    # Conv4 layer
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

    # Conv5 layer
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
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
    model.add(Dense(len(classes), activation='softmax'))

    optimizer = Adam(0.0001)

    # Using TensorBoard callback for data visualization
    tbCallback = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)

    # Using ModelCheckpoint for saving the best model overall
    modelchkp = ModelCheckpoint('models/saved_model.h5', save_best_only=True)

    # Compiling a model.
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Training a model.
    model.fit_generator(generator=train_generator, steps_per_epoch=62500//16, epochs=20,
                        validation_data=test_generator, validation_steps=12524//8,
                        callbacks=[tbCallback, modelchkp])


if __name__ == '__main__':
    main()
