"""
    Testing our model on some images.
"""

from keras.preprocessing import image
from keras.models import load_model

def main():
    # Size of images.
    img_width = 140
    img_height = 209

    model = load_model('models/saved_model.h5')

    # Load test image.
    img = image.load_img('data/cee.png', target_size=(img_width, img_height, 3))
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
