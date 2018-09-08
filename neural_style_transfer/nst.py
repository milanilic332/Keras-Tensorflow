"""
    Neural style transfer
    Used and modified code from:
    https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py

    Minimizing loss:
    loss = style_loss + content_loss (+ tv_loss)
"""
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
import time

from keras.applications import vgg19
from keras import backend as K


content_img_path = 'data/square.jpg'
style_img_path = 'data/sonny.jpg'
result_prefix = 'data/result'

iterations = 25
content_weight = 0.001
style_weight = 1.0
tv_weight = 1.0

# dimensions of the generated picture (CPU and low memory)
width, height = load_img(content_img_path).size
res_nrows = 400
res_ncols = int(width*res_nrows/height)


# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(res_nrows, res_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


# util function to convert a tensor into a valid image
def deprocess_image(x):
    x = x.reshape((res_nrows, res_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# get tensor representations of images
content_img = K.variable(preprocess_image(content_img_path))
style_img = K.variable(preprocess_image(style_img_path))

# this will contain generated image
generated_image = K.placeholder((1, res_nrows, res_ncols, 3))

# combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([content_img,
                              style_img,
                              generated_image], axis=0)

# build the VGG19 network with our 3 images
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)

# get the symbolic outputs of each "key" layer.
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


# the gram matrix of an image tensor
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# style loss function (given style and gen image)
def style_loss(style, generated):
    S = gram_matrix(style)
    C = gram_matrix(generated)
    channels = 3
    size = res_nrows * res_ncols
    return K.sum(K.square(S - C))/(4. * (channels ** 2)*(size ** 2))


# content loss function (given content and gen image)
def content_loss(base, generated):
    return (1/2)*K.sum(K.square(generated - base))


# the 3rd loss function, total variation loss, designed to keep the generated image locally coherent 
def total_variation_loss(x):
    a = K.square(x[:, :res_nrows - 1, :res_ncols - 1, :] - x[:, 1:, :res_ncols - 1, :])
    b = K.square(x[:, :res_nrows - 1, :res_ncols - 1, :] - x[:, :res_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# add up losses
loss = K.variable(0.)
layer_features = outputs_dict['block5_conv2']
content_img_features = layer_features[0, :, :, :]
generated_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(content_img_features, generated_features)

feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_features = layer_features[1, :, :, :]
    generated_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, generated_features)
    loss += (style_weight/len(feature_layers)) * sl

loss += tv_weight * total_variation_loss(generated_image)

# get the gradients of the generated image
grads = K.gradients(loss, generated_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([generated_image], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1, res_nrows, res_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')

    return loss_value, grad_values


# this Evaluator class makes it possible to compute loss and gradients in one pass
# while retrieving them via two separate functions, "loss" and "grads".
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

# using L-BFGS to minimize loss
x = preprocess_image(content_img_path)

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)

    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.jpg' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))