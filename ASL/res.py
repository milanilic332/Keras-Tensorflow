from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense
from keras.models import Model
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


    base_model = InceptionResNetV2(include_top=False, pooling='max', input_shape=(img_width, img_height, 3))
    outputs = Dense(len(classes), activation='softmax')(base_model.output)
    model = Model(base_model.inputs, outputs)

    optimizer = Adam(0.0001)

    # Compiling a model.
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Using TensorBoard callback for data visualization
    tbCallback = TensorBoard(log_dir='logs/res', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)

    # Using ModelCheckpoint for saving the best model overall
    modelchkp = ModelCheckpoint('models/res/saved_model.h5', save_best_only=True)

    # Training a model.
    model.fit_generator(generator=train_generator, steps_per_epoch=62500//16, epochs=5,
                        validation_data=test_generator, validation_steps=12524//16,
                        callbacks=[tbCallback, modelchkp])


if __name__ == '__main__':
    main()
