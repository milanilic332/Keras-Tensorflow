"""
    Testing our model on some images.
"""

from keras.preprocessing import image
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

    # Model architecture
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Load saved weights.
    model.load_weights('models/final_CNN.h5')

    # Load test image.
    img = image.load_img('data/cara.png', target_size=(img_width, img_height, 3))
    img = image.img_to_array(img)

    # Reshaping for predict method.
    test_img = img.reshape((1, img_width, img_height, 3))

    # Predicting class of test image.
    pred_class = model.predict_classes(test_img)

    # Decode class
    if pred_class[0] == [0]:
        print('male')
    else:
        print('female')


if __name__ == '__main__':
    main()
