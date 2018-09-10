# Keras-Tensorflow
Building deep neural networks using keras and tensorflow for various problems.

* ASL:  
  * Image classification of sign language letters, dataset from https://www.kaggle.com/grassknoted/asl-alphabet.  
  * Train data - 2500 images for every class.  
  * Validation data - 500 images for every class.  
  * Used image augmentation and got around 97% accuracy with my cnn.
  * With InceptionResNetV2() got 99.5%.

* IMDb faces:  
  * Image classification of actors and actresses based on gender. Around 94% accuracy.
    
* Rnn:
  * LSTM network for generating new C code.
  * Used limited number of C files from linux-kernel.
  * After 50 epochs loss came down to 0.73.
  
* neural_style_transfer:
  * Algorithm for generating image in style of another image.
