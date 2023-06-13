from __future__ import division
import os
import numpy as np
import keras
from keras.models import load_model, model_from_json
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Lambda, concatenate, UpSampling2D
from keras.layers.merge import add, concatenate, dot
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from PIL import Image
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from tensorflow.python.client import device_lib
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from tensorflow.contrib.image import dense_image_warp
from keras.optimizers import Adam
import cv2
from keras.regularizers import l2
from keras.preprocessing.image import img_to_array, load_img, save_img
from keras.utils import plot_model
import sys
import csv

custom_LeakyReLU = LeakyReLU(alpha=0.1)
custom_LeakyReLU.__name__ = 'relu'

# gpu settings, use_gpu = 0 for cpu
gpu_device = '2'
use_gpu = 1
if use_gpu:
    device_use = "/gpu:" + str(gpu_device)
else:
    device_use = '/cpu:0'

if use_gpu:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

gpu_memory = 0.95
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
config.allow_soft_placement = True
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)

if use_gpu:
    assert 'GPU' in str(device_lib.list_local_devices())

"""
=============
Start of Flow Section (code from https://github.com/Johswald/flow-code-python)
=============
"""

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8
TAG_FLOAT = 202021.25
LR = 0.0001

def read_flow(file):
    #assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == ".flo", "file ending is not .flo %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count = 1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count = 1)
    h = np.fromfile(f, np.int32, count =1)
    try:
        data = np.fromfile(f, np.float32, count = 2*w*h)
    except:
        data = np.fromfile(f, np.float32, count = 2*w[0]*h[0])
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

TAG_STRING = 'PIEH'
def write_flow(flow, filename):
    try:
        assert type(filename) is str, "file is not str %r" % str(filename)
    except:
        if type(str(filename)) is str:
            filename = str(filename)
        else:
            assert type(filename) is str, "file is not str %r" % str(filename)
    assert filename[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    try:
        channel_3 = flow.shape[3]
        flow = flow[0]
        height, width, nBands = flow.shape
        exception_counter = 0
    except:
        try:
            height, width, nBands = flow.shape
            exception_counter = 0
        except:
            flow_temp = np.array(flow)
            height, width, nBands = flow_temp.shape
            exception_counter = 1

    assert nBands == 2, "number of bands = %r != 2" % nBands
    if exception_counter:
        u = flow_temp[:, :, 0]
        v = flow_temp[:, :, 1]
    else:
        u = flow[:, :, 0]
        v = flow[:, :, 1]
    assert u.shape == v.shape, "Invalid flow shape"
    height, width = u.shape
    pyvers = sys.version_info[0]
    if pyvers < 3:
        f = open(filename, 'wb')
    elif pyvers >= 3:
        f = open(filename, 'w')
    f.write(TAG_STRING)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width)*2] = u
    tmp[:, np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def makeColorwheel():

    #  color encoding scheme

    #   adapted from the color circle idea described at
    #   http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3]) # r g b

    col = 0
    #RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY

    #YG
    colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col:YG+col, 1] = 255;
    col += YG;

    #GC
    colorwheel[col:GC+col, 1]= 255
    colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC;

    #CB
    colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col:CB+col, 2] = 255
    col += CB;

    #BM
    colorwheel[col:BM+col, 2]= 255
    colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM;

    #MR
    colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col:MR+col, 0] = 255
    return     colorwheel

def computeColor(u, v):

    colorwheel = makeColorwheel();
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v)

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)     # 1, 2, ..., ncols
    k1 = k0+1;
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1],3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius
        col[~idx] *= 0.75 # out of range
        img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

    return img.astype(np.uint8)


def computeImg(flow):

    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10

    u = flow[: , : , 0]
    v = flow[: , : , 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    #fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0
    v[greater_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v))
    maxrad = max([maxrad, np.amax(rad)])
    #print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

    u = u/(maxrad+eps)
    v = v/(maxrad+eps)
    img = computeColor(u, v)
    return img
"""
=============
End of Flow Section
=============
"""

# Custom average EPE and AAE error functions used in loss functions
def average_endpoint_error(x, y):
    if tf.rank(x).eval(session=sess) < 4:
        x = tf.expand_dims(x, 0)
        y = tf.expand_dims(y, 0)
    d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), 3))
    meand = tf.reduce_mean(d, [1,2])
    return meand

def average_angular_error(x, y):
    if tf.rank(x).eval(session=sess) < 4:
        x = tf.expand_dims(x, 0)
        y = tf.expand_dims(y, 0)
    x = tf.concat([x, tf.ones_like(x)[:, :, :, :-1]], axis=3)
    y = tf.concat([y, tf.ones_like(y[:, :, :, :-1])], axis=3)
    cross_product = tf.sqrt(tf.reduce_sum((tf.multiply(tf.linalg.cross(x, y), tf.linalg.cross(x, y))), axis=3))
    dot_product = tf.reduce_sum(tf.multiply(x,y), axis=3)
    arctan_compute = tf.abs(tf.atan2(cross_product, dot_product))
    aae = tf.reduce_mean(arctan_compute, [1, 2])
    return aae

import numpy as np

# Custom function to use either regularization, activation, both, or neither for layers
def reg_activate(filter_inf, window, stride, padding, name, reg, activ, reg_val, convolution, conv_activation):
    # for conv layers
    if convolution:
        if reg and activ:
            conv = Conv2D(filter_inf, (window, window), strides = stride, padding = padding, name=name,activation = conv_activation, kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_value))
        elif reg and not activ:
            conv = Conv2D(filter_inf, (window, window), strides = stride, padding = padding, name=name, kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))
        if activ and not reg:
            conv = Conv2D(filter_inf, (window, window), strides = stride, padding = padding, name=name,activation = conv_activation)
        else:
            conv = Conv2D(filter_inf, (window, window), strides = stride, padding = padding, name=name)
    # for deconv layers
    else:
        if reg and activ:
            conv = Conv2DTranspose(filter_inf, (window, window), strides = stride, padding = padding, name=name,activation = conv_activation, kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_value))
        elif reg and not activ:
            conv = Conv2DTranspose(filter_inf, (window, window), strides = stride, padding = padding, name=name, kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))
        elif activ and not reg:
            conv = Conv2DTranspose(filter_inf, (window, window), strides = stride, padding = padding, name=name,activation = conv_activation)
        else:
            conv = Conv2DTranspose(filter_inf, (window, window), strides = stride, padding = padding, name=name)
    return conv

#
class BilinearInterpolation(Layer):
    """Performs bilinear interpolation as a keras layer (code used from: https://github.com/oarriaga/STN.keras/tree/master)
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya
    """

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(**kwargs)

    def get_config(self):
        return {
            'output_size': self.output_size,
        }

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _interpolate(self, image, sampled_grids, output_size):

        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]

        x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')

        x = .5 * (x + 1.0) * K.cast(width, dtype='float32')
        y = .5 * (y + 1.0) * K.cast(height, dtype='float32')

        x0 = K.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(K.int_shape(image)[2] - 1)
        max_y = int(K.int_shape(image)[1] - 1)

        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)

        pixels_batch = K.arange(0, batch_size) * (height * width)
        pixels_batch = K.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = K.flatten(base)

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = K.reshape(image, shape=(-1, num_channels))
        flat_image = K.cast(flat_image, dtype='float32')
        pixel_values_a = K.gather(flat_image, indices_a)
        pixel_values_b = K.gather(flat_image, indices_b)
        pixel_values_c = K.gather(flat_image, indices_c)
        pixel_values_d = K.gather(flat_image, indices_d)

        x0 = K.cast(x0, 'float32')
        x1 = K.cast(x1, 'float32')
        y0 = K.cast(y0, 'float32')
        y1 = K.cast(y1, 'float32')

        area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = K.shape(X)[0], K.shape(X)[3]
        transformations = K.reshape(affine_transformation,
                                    shape=(batch_size, 2, 3))
        # transformations = K.cast(affine_transformation[:, 0:2, :], 'float32')
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        sampled_grids = K.batch_dot(transformations, regular_grids)
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = K.reshape(interpolated_image, new_shape)
        return interpolated_image

# Custom flow loss wrappers, to be used in the loss functions

# Wrapper of the huber loss for reconstruction error
def flow_losses(eps, alpha_1, L1):
    def robust_penalty(X, alpha):
        return K.pow(K.square(X) + K.square(eps), alpha)
    def loss(y_true, y_pred):
        if not L1:
            photometric_loss = robust_penalty(y_true - y_pred, alpha_1)
            photometric_loss = K.mean(photometric_loss, axis=-1)
        else:
            photometric_loss = tf.losses.huber_loss(y_true, y_pred)
        weighted_loss = photometric_loss
        return weighted_loss
    return loss

# Wrapper for image gradients for smoothness constraint
def smoothness_loss(eps, alpha):
    def robust_penalty(X):
        return K.pow(K.square(X) + K.square(eps), alpha)
    def loss_fcn(y_true, y_pred):
        dy, dx = tf.image.image_gradients(y_pred)
        gradients = 0
        for j in range(2):
            gradients += robust_penalty(dx[:, :, :, j])
            gradients += robust_penalty(dy[:, :, :, j])
        gradients = K.mean(gradients, axis=-1)
        return gradients
    return loss_fcn

# Bilinear upsampling for intermediate flow prediction layers
def UpSampling2DBilinear(stride, **kwargs):
    def layer(x):
        try:
            input_shape = K.int_shape(x)
        except:
            import tensorflow as tf
            import keras.backend as K
            input_shape = K.int_shape(x)
        output_shape = (int(float(input_shape[1])/float(stride)), int(float(input_shape[2])/float(stride)))
        try:
            return tf.image.resize_bicubic(x, output_shape, align_corners=True)
        except:
            import tensorflow as tf
            return tf.image.resize_bicubic(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)

# Function used to build the keras CNN model with different parameters
def getModels(Network_info, initial_weightpath):
    filters, redirect_filters, redirect_layer_info, redirect_receiver_info, regularization = Network_info
    eps, alpha_1, alpha_2, L1, weights, flow_weights = loss_params
    redirect_layers = [] # stores layers to be redirected
    print ("Generating model with height={}, width={},batch_size={}".format(height,width,batch_size))
    redirect_layer_counter, layer_counter = 0, 0
    ## activation function
    conv_activation = custom_LeakyReLU 
    # left and right inputs
    input_l = Input(shape=(height, width, 3), name='pre_input')
    input_r = Input(shape=(height, width, 3), name='nxt_input')
    print('input = '+str(input_l))
    #layer 1, output of layer 1 is height/2 x width/2
    conv1 = reg_activate(filters[0], 7, 2, 'same', 'conv1', regularize, activate, 0.01, conv_activation = conv_activation, convolution = 1)
    conv1_l = conv1(input_l)
    conv1_r = conv1(input_r)
    temporary_redirect_layer = conv1_l
    print('conv1 = ' + str(conv1_l))

    # If redirection is used, keeps this layer in redirect_layers using redirect_layer_info for later redirection 
    # in layers specified by redirect_receiver_info
    # Example:
    # redirect_layer_info = np.array([2, 3, 4, 5])
    # redirect_receiver_info = np.transpose(np.array([3, 4, 5, 6]))
    # redirect_filters = np.array([32, 32, 32, 32, 32, 32])
    # This means that conv layers 2, 3, 4, 5 will be concatenated, in the same order, with layers 3, 4, 5, 6
    
    # If no redirection is specified, the network will redirect as done in the original FlowNetS.
    # This means conv3 (saved as conv3_1) will be concatenated with deconv3 later, conv4 with deconv4, and so on
    layer_counter += 1
    if not (redirect_layer_counter == len(redirect_layer_info)):
        while(layer_counter == redirect_layer_info[redirect_layer_counter]):
            redirect_layers.append(temporary_redirect_layer)
            redirect_layer_counter += 1
            if redirect_layer_counter == len(redirect_layer_info):
                break

    #layer 2 output of layer 2 is height/4 x width/4
    conv2 = reg_activate(filters[1], 5, 2, 'same', 'conv2', regularize, activate, 0.01, conv_activation = conv_activation, convolution = 1)
    conv2_l = conv2(conv1_l)
    conv2_r = conv2(conv1_r)
    print('conv2 = '+str(conv2_l))

    # Same as above for redirection
    temporary_redirect_layer = conv2_l
    layer_counter += 1
    if not (redirect_layer_counter==len(redirect_layer_info)):
        while(layer_counter == redirect_layer_info[redirect_layer_counter]):
            redirect_layers.append(temporary_redirect_layer)
            redirect_layer_counter += 1
            if redirect_layer_counter == len(redirect_layer_info):
                break
    # Layer 2 can be a receiver info, as well as the all the layers from here onwards. This will concatenate the redirected
    # layer specified by redirect_receiver_info
    for i,info in enumerate(redirect_receiver_info):
        if info == layer_counter:
            difference = redirect_receiver_info[i] - redirect_layer_info[i]
            redirect_layers[i]=Conv2D(32,(1,1),strides=2**(difference),padding = 'same',activation=conv_activation)(redirect_layers[i])
            conv2_l = concatenate([conv2_l,redirect_layers[i]], axis = 3)

    #layer 3 output of layer 3 is height/8 x width8
    conv3 = reg_activate(filters[2], 5, 2, 'same', 'conv3', regularize, activate, 0.01,conv_activation = conv_activation, convolution = 1)
    conv3_l = conv3(conv2_l)
    conv3_r = conv3(conv2_r)
    print('conv3 = '+str(conv3_l))

    temporary_redirect_layer = conv3_l
    layer_counter += 1
    if not (redirect_layer_counter==len(redirect_layer_info)):
        while(layer_counter == redirect_layer_info[redirect_layer_counter]):
            redirect_layers.append(temporary_redirect_layer)
            redirect_layer_counter += 1
            if redirect_layer_counter == len(redirect_layer_info):
                break
    # merge
    conv3_corr = concatenate([conv3_l, conv3_r])
    # merged convolution
    conv_l_redir = Conv2D(redirect_filters[2], (1, 1), name = "conv_redir", activation = conv_activation)(conv3_l)
    conv3_corr = concatenate([conv_l_redir, conv3_corr])
    print('combined layer = ' + str(conv3_corr))
    for i,info in enumerate(redirect_receiver_info):
        if info == layer_counter:
            difference = redirect_receiver_info[i] - redirect_layer_info[i]
            redirect_layers[i] = Conv2D(32, (1, 1), strides = 2**difference, padding = 'same', activation = conv_activation)(redirect_layers[i])
            conv3_corr = concatenate([conv3_corr, redirect_layers[i]])
    conv3_1 = reg_activate(filters[2], 3, 1, 'same', 'conv3_1', regularize, activate, regularization, conv_activation = conv_activation, convolution = 1)(conv3_corr)
    print('conv3_combined = ' + str(conv3_1))
    #layer 4, output of layer 4 is height/16 x width/16
    conv4 = reg_activate(filters[3], 3, 2, 'same', 'conv4', regularize, activate, regularization, conv_activation = conv_activation, convolution = 1)(conv3_1)
    height_16, width_16 = height/16, width/16
    conv4_1 = Conv2D(filters[3], (3, 3), strides = 1, padding = 'same', name = "conv4_1", activation = conv_activation)(conv4)

    temporary_redirect_layer = conv4_1
    layer_counter += 1
    print('conv4_1 = '+str(conv4_1))
    
    if not (redirect_layer_counter == len(redirect_layer_info)):
        while(layer_counter == redirect_layer_info[redirect_layer_counter]):
            redirect_layers.append(temporary_redirect_layer)
            redirect_layer_counter += 1
            if redirect_layer_counter == len(redirect_layer_info):
                break
    for i,info in enumerate(redirect_receiver_info):
        if info == layer_counter:
            difference = redirect_receiver_info[i] - redirect_layer_info[i]
            redirect_layers[i]=Conv2D(32, (1, 1), strides = 2**(difference), padding = 'same', activation = conv_activation)(redirect_layers[i])
            conv4_1 = concatenate([conv4_1, redirect_layers[i]])

    # layer 5, now /32
    conv5 = reg_activate(filters[4], 3, 2, 'same', 'conv5', regularize, activate, 0.01, conv_activation = conv_activation, convolution = 1)(conv4_1)
    height_32, width_32 = height_16/2, width_16/2
    conv5_1 = Conv2D(filters[4], (3, 3), strides = 1, padding = 'same', name = 'conv5_1', activation = conv_activation)(conv5)
    
    temporary_redirect_layer = conv5_1
    print('conv5_1 = ' + str(conv5_1))
    layer_counter += 1
    if not (redirect_layer_counter == len(redirect_layer_info)):
        while(layer_counter == redirect_layer_info[redirect_layer_counter]):
            redirect_layers.append(temporary_redirect_layer)
            redirect_layer_counter += 1
            if redirect_layer_counter == len(redirect_layer_info):
                break
    for i,info in enumerate(redirect_receiver_info):
        if info == layer_counter:
            difference = redirect_receiver_info[i] - redirect_layer_info[i]
            redirect_layers[i] = Conv2D(32, (1, 1), strides = 2**(difference), padding = 'same', activation = conv_activation)(redirect_layers[i])
            conv5_1 = concatenate([conv5_1,redirect_layers[i]])

    # Layer 6, now /64
    conv6 = reg_activate(filters[5], 3, 2, 'same', 'conv6', regularize, activate, regularization, conv_activation = conv_activation, convolution = 1)(conv5_1)
    height_64, width_64 = height_32/2, width_32/2 
    print('conv6 = '+str(conv6))
    print ("Compiling encoder...")

    encoderModel = Model(inputs = [input_l, input_r], outputs = conv6)
    print ("Finished encoder! :)")

    conv6_1 = Conv2D(filters[5], (3, 3), padding='same', strides = 1, activation = 'relu')(conv6)

    temporary_redirect_layer = conv6
    layer_counter += 1
    
    if not (redirect_layer_counter==len(redirect_layer_info)):
        while(layer_counter == redirect_layer_info[redirect_layer_counter]):
            redirect_layers.append(temporary_redirect_layer)
            redirect_layer_counter += 1
            if redirect_layer_counter == len(redirect_layer_info):
                break
    for i,info in enumerate(redirect_receiver_info):
        if info == layer_counter:
            print('i=' + str(i))
            difference = redirect_receiver_info[i] - redirect_layer_info[i]
            redirect_layers[i]=Conv2D(32, (1, 1), strides = 2**(difference), padding = 'same', activation = conv_activation)(redirect_layers[i])
            conv6_1 = concatenate([conv6_1, redirect_layers[i]])

    # Deconvolution layers, starting with activating conv6_1
    deconv5 = reg_activate(filters[4], 4, 2, 'same', 'deconv5', regularize, activate, regularization,conv_activation = conv_activation, convolution = 0)(conv6_1)
    # Intermediate flow predictions start here, named 'predict'
    deconv6_predict = Conv2D(2, kernel_size = (3, 3), strides = 1,padding = 'same', name = 'Deconv6_prediction')(conv6_1)
    deconv6_predict_size = K.int_shape(deconv6_predict)
    # Need to upsample intermediate prediction to concatenate with next deconvolved layer and redirected layer from earlier
    upsampled_deconv6_predict = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(deconv6_predict)
    upsampled_deconv6_predict = Lambda(lambda x: x * 2.0)(upsampled_deconv6_predict)
    # Concatenate next layer with upsampled flow prediction + earlier redirected layer
    deconv5 = concatenate([conv5_1, deconv5, upsampled_deconv6_predict], name = "deconv5_concatentation")
    deconv5 = Conv2D(2*filters[4], (1, 1), strides = 1, padding = 'same', activation='relu')(deconv5)
    layer_counter += 1

    # Repeat the above process until the final full-sized flow prediction
    deconv4 = reg_activate(filters[3], 4, 2, 'same', 'deconv4', regularize, activate, regularization, conv_activation = conv_activation,convolution = 0)(deconv5)
    deconv5_predict = Conv2D(2, kernel_size=(3, 3), strides = 1, padding = 'same', name = 'Deconv5_prediction')(deconv5)
    deconv5_predict_size = K.int_shape(deconv5_predict)
    upsampled_deconv5_predict = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(deconv5_predict)
    upsampled_deconv5_predict = Lambda(lambda x: x * 2.0)(upsampled_deconv5_predict)
    deconv4 = concatenate([conv4_1 , deconv4, upsampled_deconv5_predict], name = "deconv4_concatentation")

    deconv4 = Conv2D(2*filters[3], (1,1), strides = 1, padding = 'same', activation = 'relu')(deconv4)
    layer_counter += 1
    deconv3 = reg_activate(filters[2], 4, 2, 'same', 'deconv3', regularize, activate, regularization,conv_activation = conv_activation, convolution = 0)(deconv4)
    deconv4_predict = Conv2D(2, kernel_size=(3, 3),  strides = 1, padding = 'same', name = 'Deconv4_prediction')(deconv4)
    deconv4_predict_size = K.int_shape(deconv4_predict)
    upsampled_deconv4_predict = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(deconv4_predict)
    upsampled_deconv4_predict = Lambda(lambda x: x * 2.0)(upsampled_deconv4_predict)
    deconv3 = concatenate([conv3_1, deconv3, upsampled_deconv4_predict],name = "deconv3_concatentation")
    deconv3 = Conv2D(2*filters[2], (1,1), padding = 'same', strides = 1, activation='relu')(deconv3)
    layer_counter += 1
    
    deconv2 = reg_activate(filters[1], 4, 2, 'same', 'deconv2', regularize, activate, regularization, conv_activation = conv_activation, convolution = 0)(deconv3)
    deconv3_predict = Conv2D(2, kernel_size = (3,3), strides = 1, padding = 'same', name = 'Deconv3_prediction')(deconv3)
    deconv3_predict_size = K.int_shape(deconv3_predict)
    upsampled_deconv3_predict = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(deconv3_predict)
    upsampled_deconv3_predict = Lambda(lambda x: x * 2.0)(upsampled_deconv3_predict)
    deconv2 = concatenate([conv2_l, deconv2, upsampled_deconv3_predict], name = "deconv2_concatenation")
    
    deconv1 = reg_activate(filters[0], 4, 2, 'same', 'deconv1', regularize, activate, regularization,conv_activation = conv_activation, convolution = 0)(deconv2)
    deconv2_predict = Conv2D(2, kernel_size = (3,3), strides = 1, padding = 'same', name = 'Deconv2_prediction')(deconv2)
    deconv2_predict_size = K.int_shape(deconv2_predict)
    upsampled_deconv2_predict = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(deconv2_predict)
    upsampled_deconv2_predict = Lambda(lambda x: x * 2.0)(upsampled_deconv2_predict)
    deconv1 = concatenate([conv1_l, deconv1, upsampled_deconv2_predict], name = "deconv1_concatenation")

    # PREDICT THE OUTPUT
    Predict_flow = Conv2D(2, kernel_size = (3, 3), strides = 1, padding = 'same', name = 'Deconv1_prediction')
    decoder_output = Predict_flow(deconv1)
    Resize_2 = K.int_shape(decoder_output)

    print("Compiling decoder...")
    # If reconstruction error is included in the model, include image warping to compute the loss between X1 warped
    # by predicted flow and X2
    if warped_model:
        upsampled_output = decoder_output
        upsample_shape = K.int_shape(upsampled_output)
        # Adjust the dimensions of decoder output (predicted flow Y_hat) and input_l (input image X1)
        reverse_layer = Lambda(lambda x: K.reverse(x, axes=3),output_shape=(upsample_shape))(upsampled_output)
        downsampled_input = UpSampling2DBilinear(2, name='input_downsample', input_shape=K.int_shape(input_l))(input_l)
        # use dense_image_warp as a Lambda layer to warp downsampled_input using decoder_output
        try:
            warped_output = Lambda(lambda inp: dense_image_warp(inp[0], inp[1]), name='warped_output_first')([downsampled_input, reverse_layer])
        except:
            import tensorflow as tf
            warped_output = Lambda(lambda inp: dense_image_warp(inp[0], inp[1]), name='warped_output_first')([downsampled_input, reverse_layer])
        warped_output = UpSampling2D(size=(2,2), interpolation='bilinear', name='warped_output')(warped_output) # Adjust dimensions
        print("output shape is : ", K.int_shape(warped_output))

    # Adjust outputs depending on experiment
    if warped_model:
        decoderModel = Model(inputs = [input_l, input_r], outputs = [decoder_output, deconv5_predict, deconv4_predict, deconv3_predict, deconv2_predict, warped_output], name='decoder')
    else:
        decoderModel = Model(inputs = [input_l, input_r], outputs = [decoder_output, deconv5_predict, deconv4_predict, deconv3_predict, deconv2_predict], name='decoder')
    # Initialize weights from pretrained model if needed
    if weights_initialize:
        decoderModel.load_weights(initial_weightpath)
    print("Finished decoder! :)")
    optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

    # Define loss dictionary, which assigns loss functions to the relevant layers, and loss weights
    if warped_model and angular_error:
        loss_dict = {'Deconv1_prediction': average_endpoint_error, 'warped_output': flow_losses(eps, alpha_1, L1),
                    'Deconv1_prediction': average_angular_error, 'Deconv2_prediction': average_endpoint_error,
                    'Deconv3_prediction': average_endpoint_error, 'Deconv4_prediction': average_endpoint_error, 
                    'Deconv5_prediction': average_endpoint_error}
        lossWeights={"Deconv1_prediction": weights[0], 'warped_output': weights[1], 'Deconv1_prediction': weights[2],
                    'Deconv2_prediction': flow_weights[0], 'Deconv3_prediction': flow_weights[1] ,'Deconv4_prediction': flow_weights[2],
                    'Deconv5_prediction': flow_weights[3]}
    elif warped_model and smoothness_on:
        loss_dict = {'Deconv1_prediction': average_endpoint_error, 'warped_output': flow_losses(eps, alpha_1, L1),
                    'Deconv1_prediction': smoothness_loss(eps, alpha_2), 'Deconv2_prediction': average_endpoint_error,
                    'Deconv3_prediction': average_endpoint_error, 'Deconv4_prediction': average_endpoint_error, 
                    'Deconv5_prediction': average_endpoint_error}
        lossWeights={"Deconv1_prediction": weights[0], 'warped_output': weights[1], 'Deconv1_prediction': weights[2],
                    'Deconv2_prediction': flow_weights[0], 'Deconv3_prediction': flow_weights[1] ,'Deconv4_prediction': flow_weights[2], 
                    'Deconv5_prediction': flow_weights[3]}
    elif warped_model:
        loss_dict = {'Deconv1_prediction': average_endpoint_error, 'warped_output': flow_losses(eps, alpha_1, L1),
                    'Deconv2_prediction': average_endpoint_error, 'Deconv3_prediction': average_endpoint_error,
                    'Deconv4_prediction': average_endpoint_error, 'Deconv5_prediction': average_endpoint_error}
        lossWeights={"Deconv1_prediction": weights[0], 'warped_output': weights[1],
                    'Deconv2_prediction': flow_weights[0], 'Deconv3_prediction': flow_weights[1] ,'Deconv4_prediction': flow_weights[2], 
                    'Deconv5_prediction': flow_weights[3]}
    else:
        loss_dict={'Deconv1_prediction': average_endpoint_error, 'Deconv2_prediction': average_endpoint_error,
                    'Deconv3_prediction': average_endpoint_error, 'Deconv4_prediction': average_endpoint_error, 
                    'Deconv5_prediction': average_endpoint_error}
        lossWeights={"Deconv1_prediction": weights[0],
                    'Deconv2_prediction': flow_weights[0], 'Deconv3_prediction': flow_weights[1], 'Deconv4_prediction': flow_weights[2], 
                    'Deconv5_prediction': flow_weights[3]}
    # Compile model and return data of the sizes
    decoderModel.compile(optimizer = optimizer, loss = loss_dict, loss_weights = lossWeights, metrics = ["acc"])
    resize_data = np.array([Resize_2, deconv6_predict_size, deconv5_predict_size, deconv4_predict_size, deconv3_predict_size, deconv2_predict_size])
    return encoderModel, decoderModel, resize_data, input_l, input_r

def Resize_image(images, Resize, origs, im_or_flow, Reshape_option = True):
    """
    =============
    This function resizes the input image H x W x D (resp. tensor of size B x H x W x D) and returns resized image (resp. tensor)
    If im_or_flow is input as '1', then the input image is actually a flow field, and the resizing will also scale the flow vectors
    based on the new dimensions.

    Arguments:
    images: input images/flow fields (size H x W x D or B x H x W x D for images, H x W or B x H x W for flow)
    Resize: tuple (H, W) of dimensions of output resized image
    origs: tuple (H, W) of dimensions of input image before resizing
    im_or_flow: set to 0 for images, 1 for flow fields

    Example: input flow fields (images) have size 1024x1024, to be resized to 512x384
    output dimension will be 512x384, and the uv components of the flow field will be rescaled as follows:
    u_new = u * 512/1024, v_new = v * 384/1024. This will keep the vector values consistent with the new dimensions.
    =============
    """

    H_orig, W_orig = origs[0], origs[1]
    if Reshape_option:
        if images.shape[2] <= 3:
            tensor = 0
            Test = images
            W = Resize
            height, width, depth = Test.shape
            imgScale = np.divide(W, np.array([width, height]))
            newX, newY = width * imgScale[0], height * imgScale[1]
        else:
            tensor = 1
            Test = images[0,:,:,:]
            W = Resize
            height, width, depth = Test.shape
            imgScale = np.divide(W, np.array([height, width]))
            newX,newY = Test.shape[1] * imgScale[1], Test.shape[0] * imgScale[0]
        if tensor:
            m = len(images[:,0,0,0])
            resized_images = np.zeros((m,Resize[0],Resize[1],depth))
            for k in range(m):
                Test = images[k,:,:,:]
                Test = cv2.resize(Test, (newX, newY))
                resized_images[k,:,:]=Test
        else:
            if im_or_flow:
                Test[:, :, 0] *= np.float(Resize[0]) / np.float(H_orig)
                Test[:, :, 1] *= np.float(Resize[1]) / np.float(W_orig)
            resized_images = cv2.resize(Test, (Resize[1], Resize[0]))
    else:
        resized_images = images
    return resized_images

def Resize_image_withlandmarks(images, Resize, landmarks):
    try:
        Test = images[0, :, :, :]
    except:
        Test = images
    height, width, depth = Test.shape
    lm_x, lm_y = landmarks[:, 0], landmarks[:, 1]
    new_lm_x, new_lm_y = float(float(Resize[1])/float(width)) * lm_x, float(float(Resize[0])/float(height)) * lm_y
    new_lm = np.c_[new_lm_x, new_lm_y]
    try:
        m = len(images[:, 0, 0, 0])
    except:
        m = 1
    resized_images = np.zeros((m,Resize[0],Resize[1],depth))
    for k in range(m):
        try:
            Test = images[k, :, :, :]
        except:
            Test = images
        Test = cv2.resize(Test, (Resize[1], Resize[0]))
        resized_images[k, :, :] = Test
    return resized_images, new_lm

def get_before_extension(filename):
        new_name = ''
        for i, letters in enumerate(filename):
                if letters == '.':
                        break
                new_name = new_name + filename[i]
        return new_name

def get_input(path):
    img = img_to_array(load_img(path))
    return(img)

def get_next_image(shuffled_list, image_list):
    """
    =============
    This function finds the second input image using the name of the first input image.
    =============
    """
    image_list_2 = []
    for i, image in enumerate(shuffled_list):
        label = image.split('.')[0]
        next_index = image_list.index(label + '.jpg') + 1
        try:
            image_list_2.append(image_list[next_index])
        except:
            image_list_2.append(image_list[next_index - 1])
            shuffled_list[i] = image_list[next_index - 2]
    return shuffled_list, image_list_2
img_formats = ['jpg', 'jpeg', 'png','bmp', 'jp2', 'webp', 'pbm', 'pgm', 'ppm', 'pxm', 'tiff', 'tif', 'hdr', 'pic']

def data_gen_upsample(img_folder, Resize_info, batch_size):
    """
    =============
    Custom data generator for use during training/validation.
    img_folder: path to directory of all images
    Resize_info: list of size dimensions of all inputs and outputs of the model
    batch_size: size of each batch
    =============
    """
    while(True):
        # get list of all images
        all_files = os.listdir(img_folder)
        all_ims = sorted([pics for pics in all_files if pics.split('.')[-1] in img_formats])
        N = len(all_ims)
        shuffle_images = sorted([pics for pics in all_files if pics.split('.')[-1] in img_formats])

        # Load all the data batchwise
        for offset in range(0, N, batch_size):
            # X1 batch: 16 images
            X1_batch_orig = shuffle_images[offset:offset + batch_size]
            # Use the names of the 16 X1 images to get the corresponding 16 X2 images
            X1_batch, X2_batch = get_next_image(X1_batch_orig, all_ims)

            # Initialize images and intermediate flow predictions using Resize info obtained earlier
            X1 = np.zeros((batch_size, Resize[0], Resize[1], 3)).astype('float')
            X2 = np.zeros((batch_size, Resize[0], Resize[1], 3)).astype('float')
            Y1 = np.zeros((batch_size, Resize_info[0][1], Resize_info[0][2], 2)).astype('float')
            Y2 = np.zeros((batch_size, Resize_info[1][1], Resize_info[1][2], 2)).astype('float')
            Y3 = np.zeros((batch_size, Resize_info[2][1], Resize_info[2][2], 2)).astype('float')
            Y4 = np.zeros((batch_size, Resize_info[3][1], Resize_info[3][2], 2)).astype('float')
            Y5 = np.zeros((batch_size, Resize_info[4][1], Resize_info[4][2], 2)).astype('float')
            Y6 = np.zeros((batch_size, Resize_info[5][1], Resize_info[5][2], 2)).astype('float')
            # loop over the batch names and preprocess images and flow files to compute X1, X2, and Y1-Y6
            for i, x1_batch in enumerate(X1_batch):
                label = x1_batch.split('.')[0]
                try:
                    # If the image has been preprocessed already and stored in the directory 'preprocessed', then it 
                    # will immediately read the flows and images from there.
                    pre_dir = img_folder + '/preprocessed/'
                    Y1[i], Y2[i], Y3[i], Y4[i], Y5[i], Y6[i] = read_flow(pre_dir + label + '_1.flo'), read_flow(pre_dir + label + '_2.flo'), read_flow(pre_dir + label + '_3.flo'), read_flow(pre_dir + label + '_4.flo'), read_flow(pre_dir + label + '_5.flo'), read_flow(pre_dir + label + '_6.flo')
                    X1[i], X2[i] = get_input(pre_dir + x1_batch), get_input(pre_dir + X2_batch[i])
                except:
                    # If the images had not been preprocessed before, it will do this now.

                    # Read original image and flow file
                    x1_orig = get_input(img_folder + '/' + x1_batch)
                    x2_orig = get_input(img_folder + '/' + X2_batch[i])
                    y_orig = read_flow(img_folder + '/' + label + '.flo')

                    # Preprocess flow: this will crop the flow fields using keypoint information and resize them if necessary.
                    y1, x1, crop_dims = preprocess_flow(x1_orig, y_orig, x1_batch, Resize = Resize, Resize_2 = Resize_info[0][1:3], crop_dims = [], first_flow = 1)
                    x2 = preprocess_flow(x2_orig, y_orig, x1_batch, Resize = Resize, Resize_2 = Resize_info[0][1:3], crop_dims = crop_dims, first_flow = 0)
                    y2, _, _ = preprocess_flow(x1_orig, y_orig, x1_batch, Resize = Resize, Resize_2 = Resize_info[1][1:3], crop_dims = [], first_flow = 1)
                    y3, _, _ = preprocess_flow(x1_orig, y_orig, x1_batch, Resize = Resize, Resize_2 = Resize_info[2][1:3], crop_dims = [], first_flow = 1)
                    y4, _, _ = preprocess_flow(x1_orig, y_orig, x1_batch, Resize = Resize, Resize_2 = Resize_info[3][1:3], crop_dims = [], first_flow = 1)
                    y5, _, _ = preprocess_flow(x1_orig, y_orig, x1_batch, Resize = Resize, Resize_2 = Resize_info[4][1:3], crop_dims = [], first_flow = 1)
                    y6, _, _ = preprocess_flow(x1_orig, y_orig, x1_batch, Resize = Resize, Resize_2 = Resize_info[5][1:3], crop_dims = [], first_flow = 1)
                    X1[i] = x1
                    X2[i] = x2
                    Y1[i], Y2[i], Y3[i], Y4[i], Y5[i], Y6[i] = rotate_flow_2(y1), rotate_flow_2(y2), rotate_flow_2(y3), rotate_flow_2(y4), rotate_flow_2(y5), rotate_flow_2(y6)
                    # Make preprocessed directory and store preprocessed images and flow fields there to prevent doing this every time.
                    mkdir_p(img_folder + '/preprocessed')
                    write_flow(Y1[i], img_folder + '/preprocessed/' + label + '_1.flo')
                    write_flow(Y2[i], img_folder + '/preprocessed/' + label + '_2.flo')
                    write_flow(Y3[i], img_folder + '/preprocessed/' + label + '_3.flo')
                    write_flow(Y4[i], img_folder + '/preprocessed/' + label + '_4.flo')
                    write_flow(Y5[i], img_folder + '/preprocessed/' + label + '_5.flo')
                    write_flow(Y6[i], img_folder + '/preprocessed/' + label + '_6.flo')
                    save_img(img_folder + '/preprocessed/' + x1_batch, X1[i])
                    save_img(img_folder + '/preprocessed/' + X2_batch[i], X2[i])
            # Yield flow fields and input images for regular model, and the same + deformed image if using reconstruction error
            if warped_model:
                yield [X1, X2], [Y1, Y3, Y4, Y5, Y6, X2]
            else:
                yield [X1, X2], [Y1, Y3, Y4, Y5, Y6]

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def create_sub_dict():
    sub_lists = []
    dict_nums = ['F001', 'F002', 'F004', 'F005', 'M001', 'M002', 'M003', 'M004', 'F003', 'F006', 'F007','F008','F009', 'F010', 'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020', 'F021', 'F022', 'F023', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018']
    for i in range(len(dict_nums)):
        sub_lists.append("S" + str(i))
    keys = zip(sub_lists, dict_nums)
    subject_dicts = dict(keys)
    return subject_dicts

def read_csv_data(csv_files):
    float_data = []
    string_data = []
    with open(csv_files) as csv_file:
        for j, row in enumerate(csv.reader(csv_file)):
            if j == 0:
                for elements in row:
                    string_data.append(elements)
            else:
                data = row
                for elements in data:
                    float_data.append(float(elements))
    return string_data, float_data
def get_csv_landmarks(string_data, float_data, eye_num):
    x_lm = np.zeros(eye_num + 1 + 68)
    y_lm = np.zeros(eye_num + 1 + 68)
    e_x_0 = string_data.index(" eye_lmk_x_0")
    e_x_f = string_data.index(" eye_lmk_x_55")

    x_lm[0:eye_num] = float_data[e_x_0:e_x_0 + eye_num]

    e_y_0 = string_data.index(" eye_lmk_y_0")
    e_y_f = string_data.index(" eye_lmk_y_55")

    y_lm[0:eye_num] = float_data[e_y_0:e_y_0 + eye_num]

    x_0 = string_data.index(" x_0")
    x_f = string_data.index(" x_67")
    y_0 = string_data.index(" y_0")
    y_f = string_data.index(" y_67")

    x_lm[eye_num + 1:] = float_data[int(x_0): int(x_f)+ 1]
    y_lm[eye_num + 1:] = float_data[int(y_0): int(y_f)+1]
    landmarks = np.c_[np.array(y_lm), np.array(x_lm)]
    origin = np.where(landmarks == [0, 0])
    for origins in origin:
        landmarks = np.delete(landmarks, origins, axis = 0)
    return landmarks

def rotate_flow(flow):
    R = np.array([[-1, 0], [0, -1]])
    R2 = np.array([[0, -1], [1, 0]])
    rotated_flow = np.matmul(flow, R)
    rotated_flow = np.matmul(rotated_flow, R2)
    rotated_flow[:, :, 0] = - rotated_flow[:, :, 0]
    return rotated_flow

def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))
    return x, y

def preprocess_flow(image, flow, label, Resize, Resize_2, crop_dims, first_flow):
    """
    =============
    preprocess_flow is used to prepare the flow fields and images to feed into the model.
    It will crop the image around the faces using the keypoints.
    =============
    """
    flow = rotate_flow(flow)
    H, W, _ = image.shape
    origs = [H, W]

    subject_label = label.split('.')[0].split('_')[0]
    sequence_label = label.split('.')[0].split('_')[1]

    csv_path = generated_dataset_path + subject_label + '/' + sequence_label + '/'
    csv_files = csv_path + subject_label + '_' + sequence_label + '_' + label.split('.')[0].split('_')[-1] + '.csv'
    string_data, float_data = read_csv_data(csv_files)
    landmarks = get_csv_landmarks(string_data, float_data, eye_num = 0)
    x_lm, y_lm = landmarks.T[1, :], landmarks.T[0, :]

    # If you have your own .txt files with landmarks, comment the five lines above
    # and uncomment the next five lines to get landmarks

    #landmarks_file = landmarks_path + subject_label + '/' + sequence_label + '/' + label.split('.')[0] + '_landmarks.txt'
    #lm_x, lm_y = Read_Two_Column_File(landmarks_file)
    #lms = np.c_[lm_x, lm_y]
    #image, landmarks = Resize_image_withlandmarks(image, np.array([512, 384]), lms)
    #x_lm, y_lm = landmarks[:, 0], landmarks[:, 1]

    if first_flow:
        r_min, r_max = np.min(y_lm), np.max(y_lm)
        c_min, c_max = np.min(x_lm), np.max(x_lm)
    else:
        r_min, r_max, c_min, c_max = crop_dims
    
    x_off_l, x_off_u, y_off_l, y_off_u = 10, 10, 10, 10
    where_are_NaNs = np.isnan(flow)
    flow[where_are_NaNs] = 0
    flow[int(r_min) - y_off_l:int(r_min), :] = 0
    flow[int(r_max):int(r_max) + y_off_u, :] = 0
    flow[:, int(c_min) - x_off_l:int(c_min)] = 0
    flow[:, int(c_max):int(c_max) + x_off_u] = 0
    color_flow = flow[max(int(r_min) - y_off_l, 0):min(int(r_max) + y_off_u, H), max(int(c_min) - x_off_l, 0):min(int(c_max) + x_off_u, W), :]
    color_im = image[max(int(r_min) - y_off_l, 0):min(int(r_max) + y_off_u, H), max(int(c_min) - x_off_l, 0):min(int(c_max) + x_off_u, W), :]
    color_flow = Resize_image(color_flow, Resize_2, origs = origs, im_or_flow = 1)
    color_im = Resize_image(color_im, Resize, origs = origs, im_or_flow = 0)
    if first_flow:
        return color_flow, color_im, [r_min, r_max, c_min, c_max]
    else:
        return color_im

def rotate_flow_2(flow):
    if flow.ndim > 3:
        flow = flow[0, :, :, :]
    R = np.array([[-1, 0], [0, -1]]).T
    R2 = np.array([[0, -1], [1, 0]]).T
    rotated_flow = np.matmul(flow, R2)
    rotated_flow = np.matmul(rotated_flow, R)
    rotated_flow[:, :, 0] = - rotated_flow[:, :, 0]
    return rotated_flow

def first(iterable, condition):
    return next(x for x in iterable if condition(x))

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
    def on_epoch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))

def get_weights(modelname, high_frac, F_flow, weight_lambdas):
    lambda_1, lambda_2, lambda_3, lambda_4 = weight_lambdas
    M = 4
    flow_weight_total = F_flow * lambda_1
    lambda_1 *= (1 - F_flow)
    
    if modelname in ['exp2I', 'exp2II']:
        weights = [lambda_1, lambda_2]
    elif modelname == 'exp3I':
        weights = [lambda_1, lambda_2, lambda_4]
    elif modelname == 'exp3II':
        weights = [lambda_1, lambda_2, lambda_3]
    else:
        weights = [lambda_1]
    
    A = np.array([[1, 0], [M, 0.5 * M * (M - 1)]])
    ws = flow_weight_total * np.matmul(np.linalg.inv(A), np.transpose(np.array([high_frac, 1])))
    flow_weights = np.zeros((M, 1))
    order = 1
    j_m = np.zeros((order+1, 1))
    for j in range(M):
        for k in range(order+1):
            j_m[k] = np.power(j, k)
        flow_weights[j] = np.matmul(ws, j_m)
    return weights, flow_weights

if __name__ == '__main__':
    # FOR THE USER TO MODIFY
    repo_path = os.getcwd()
    results_path = repo_path + '/OFCNN_evaluation/' # name and location of path to store the results in
    mkdir_p(results_path)

    BP4D_data_root = '/hdd2/malkaddour/datasets/BP4D/' # Change this to the root path of your dataset
    generated_dataset_path = BP4D_data_root + 'generated_dataset/' # path to generated optical flow dataset
    single_directory_path = BP4D_data_root + 'single_directory/' # path to folder with all images in a single directory

    # This is the path to your landmarks if you did not generate landmarks with our method in facialof_datasetgen.py
    # You will also need to uncomment reading the landmarks in the 'preprocess_flow' function (details specified there)
    landmarks_path = '/hdd2/malkaddour/datasets/CK/CK+/CK+/Landmarks/' 

    file_list_train, file_list_val  = single_directory_path + 'train', single_directory_path + 'val'
    file_list_test = single_directory_path + 'test'

    modelname = "exp2II" # Change this to one of exp1, exp2I, exp2II, exp3I, exp3II to build the appropriate model and loss

    # Weight paths
    weights_initialize = True # Set to False if you want to train without initializing weights
    weights_ext = 'cp.ckpt' # Name of the pretrained weights file
    weights_path = repo_path + '/exp2/caseI/' + weights_ext # path to pretrained weights

    batch_size = 16
    N_train_max, N_val_max = int(np.floor(len(os.listdir(file_list_train))/2)) - 1, int(np.floor(len(os.listdir(file_list_val))/2)) - 1
    TrainingNumber, ValNumber = 100000000, 10000000
    TrainingNumber, Valnumber = min(N_train_max, TrainingNumber), min(N_val_max, ValNumber) # If training/val are too high, cap at maximum
    num_epochs = 15 # change to number of epochs

    height, width = 384, 512 # size of the input images

    activate, regularize = True, False # settings for the model
    regularization = 0.001 # value of regularization

    # Redirection configuration
    # If redirecting is True, redirects layers in layer_info to those in receiver_info when building the model
    # Example:
    # redirect_layer_info = np.array([2, 3, 4, 5])
    # redirect_receiver_info = np.transpose(np.array([3, 4, 5, 6]))
    # This means that conv layers 2, 3, 4, 5 will be concatenated, in the same order, with layers 3, 4, 5, 6
    # You can also add redirect_filters to convolve layers before concatenation.
    # If "redirecting" is set to False, this will not use any redirection
    
    redirecting = False
    if redirecting:
        redirect_layer_info = np.array([2, 3, 4, 5])
        redirect_receiver_info = np.transpose(np.array([3, 4, 5, 6]))
    else:
        redirect_layer_info = []
        redirect_receiver_info = []
    
    redirect_filters = np.array([32, 32, 32, 32, 32, 32])
    filters = np.array([64, 128, 256, 512, 512, 1024]) # filter sizes for CNN
    
    Steps = 1
    Resize, Resize_tuple = np.array([height, width]), (height, width, 3)

    # parameters for loss functions
    alpha_1, alpha_2, loss_eps = 0.50, 0.50, 0.01
    F_flows, high_frac = 1.0/6.0, 0.50

    # Model settings based on desired model
    if modelname == "exp2I":
        EPE_lambda, CYCLIC_lambda, AAE_lambda, SMOOTHNESS_lambda = 0.4, 0.6, 0, 0
        warped_model, L1_loss, angular_error, smoothness_on = True, True, False, False
    if modelname == "exp2II":
        EPE_lambda, CYCLIC_lambda, AAE_lambda, SMOOTHNESS_lambda = 0.75, 0.25, 0, 0
        warped_model, L1_loss, angular_error, smoothness_on = True, True, False, False
    elif modelname == 'exp3I':
        EPE_lambda, CYCLIC_lambda, AAE_lambda, SMOOTHNESS_lambda = 0.3, 0.4, 0, 0.3
        warped_model, L1_loss, angular_error, smoothness_on = True, True, False, True
    elif modelname == 'exp3II':
        EPE_lambda, CYCLIC_lambda, AAE_lambda, SMOOTHNESS_lambda = 0.3, 0.4, 0.3, 0
        warped_model, L1_loss, angular_error, smoothness_on = True, True, True, False
    else:
        EPE_lambda, CYCLIC_lambda, AAE_lambda, SMOOTHNESS_lambda = 1, 0, 0, 0
        warped_model, L1_loss, angular_error, smoothness_on = True, True, False, False
    
    # Compute fractions of lambda_1 using get_weights (intermediate flow predictions)
    weight_lambdas = [EPE_lambda, CYCLIC_lambda, AAE_lambda, SMOOTHNESS_lambda]
    weights, flow_weights = get_weights(modelname, high_frac, F_flows, weight_lambdas)
    loss_params = [loss_eps, alpha_1, alpha_2, L1_loss, weights, flow_weights]

    # Store model configuration
    Network_info = [filters, redirect_filters, redirect_layer_info, redirect_receiver_info, regularization]

    with tf.device(device_use):
        # Obtain models using above configurations
        encoderModel, decoderModel, resize_data, input_l, input_r = getModels(Network_info = Network_info, initial_weightpath = weights_path)
        LearningRate = 0.01/10000
        optimizer = Adam(lr = LearningRate, beta_1 = 0.9, beta_2 = 0.999)

        # Custom data generator to preprocess the batch inputs
        image_generator_train = data_gen_upsample(file_list_train, resize_data, batch_size)
        image_generator_val = data_gen_upsample(file_list_val, resize_data, batch_size)

        loss_history = LossHistory()
        iterations = Steps * TrainingNumber / (batch_size)
        iterations_val = Steps * ValNumber / (batch_size)

        # Custom learning rate scheduler
        def step_decay(epoch):
            lambda1 = LearningRate/4
            return lambda1
        lrate = LearningRateScheduler(step_decay)

        # Configure callbacks, weights will be stored in results path configured earlier
        callbacks_list = [loss_history, lrate]
        cp_callback = keras.callbacks.ModelCheckpoint(filepath = results_path + '/cp.ckpt',
                                                    save_weights_only = True, verbose = 1)
        callbacks_list.append(cp_callback)

        # Train the model
        history = decoderModel.fit_generator(image_generator_train, epochs = num_epochs, validation_data = image_generator_val,
                                            steps_per_epoch = iterations, validation_steps = iterations_val, verbose = 1, 
                                            callbacks = callbacks_list)