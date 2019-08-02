from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import BatchNormalization, Activation, Conv2D, Lambda, TimeDistributed,\
    Add, AveragePooling2D, Dense, Flatten
import tensorflow as tf
from Layers.FixedBatchNormalization import FixedBatchNormalization
from Layers.DepthwiseConv2D import DepthwiseConv2D
from Layers.ROIPooling import ROIPooling
from Helpers.config import Config
from tensorflow.python import keras

conv_has_bias = True
W_regularizer = None
init_ = 'glorot_uniform'

Config = Config()

def _depthwise_conv_block_classification(inputs, pointwise_conv_filters, alpha,
                                         depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(tf.nn.relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(tf.nn.relu6, name='conv_pw_%d_relu' % block_id)(x)

def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    # conv block time distributed
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3


    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2x')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis, name="fixed_batch_normalisation_2x"),  name=bn_name_base + '2xt')(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2y')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis, name="fixed_batch_normalisation_2y"), name=bn_name_base + '2yt')(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2yt', trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization( axis=bn_axis, name="fixed_batch_normalisation_z"), name=bn_name_base + '2zt')(x)
    shortcut = TimeDistributed(Conv2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis, name="fixed_batch_normalisation_112"), name=bn_name_base + '112')(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def _depthwise_conv_block_detection(input, layer_name, strides = (1,1),
                          kernel_size = 3,
                          pointwise_conv_filters=32, alpha=1.0, depth_multiplier=1,
                          padding = 'valid',
                          data_format = None,
                          activation = None, use_bias = True,
                          depthwise_initializer='glorot_uniform',
                          pointwise_initializer='glorot_uniform', bias_initializer = "zeros",
                          bias_regularizer= None, activity_regularizer = None,
                          depthwise_constraint = None, pointwise_constraint = None,
                          bias_constraint= None, batch_size = None,
                          block_id=1,trainable = None, weights = None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((kernel_size, kernel_size),
                        padding=padding,
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name=layer_name + '_conv_dw_%d' % block_id)(input)
    x = BatchNormalization(axis=channel_axis, name=layer_name + '_conv_dw_%d_bn' % block_id)(x)
    x = Activation(tf.nn.relu6, name=layer_name+'_conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding=padding,
               use_bias=False,
               strides=(1, 1),
               name=layer_name + '_conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,  name=layer_name+'_conv_pw_%d_bn' % block_id)(x)
    return Activation(tf.nn.relu6,  name=layer_name+ '_conv_pw_%d_relu' % block_id)(x)



def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(tf.nn.relu, name='conv1_relu')(x)


def bn_conv(input_layer, layer_name, nb_filter, nb_row, nb_col, subsample =(1,1), border_mode ='same', bias=conv_has_bias):
    tmp_layer = input_layer
    tmp_layer = Conv2D(nb_filter, nb_row, nb_col,  activation=None, padding=border_mode, name=layer_name, use_bias=bias, kernel_regularizer=W_regularizer)(tmp_layer)
    tmp_layer = BatchNormalization(name = layer_name + '_bn')(tmp_layer)
    tmp_layer = Lambda(lambda x:tf.nn.relu(x), name=layer_name + '_nonlin')(tmp_layer)
    return tmp_layer



def light_head(input, kernel=15, padding="valid", bias=False, bn=False):
    k_width = (kernel, 1)
    k_height = (1, kernel)
    x = keras.layers.Conv2D(256, kernel_size=k_width, strides=1,  use_bias=bias, padding=padding)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(490, kernel_size=k_height, strides=1, use_bias=bias, padding=padding)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, kernel_size=k_width, strides=1, use_bias=bias, padding=padding)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(490, kernel_size=k_height, strides=1, use_bias=bias, padding=padding)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(tf.nn.relu)(x)

    return x

def rpn(image, base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    return keras.Model(image, [x_class, x_regr], name="rpn_model")


def pose(base_layers, rpn, pool_size, num_classes, train_bn=True):
    """
    inspired by https://github.com/kemangjaka/Deep-6dPose/blob/44841f4f428b679752ef52014cb06364385d06ff/mrcnn/model.py
    :param base_layers:
    :param rpn:
    :param pool_size:
    :param num_classes:
    :param train_bn:
    :return:
    """

    # roipooling before passing.
    x = keras.layers.TimeDistributed(Conv2D(1024, (pool_size, pool_size), padding="valid"), name="frcnn_pose_conv1")
    x = keras.layers.TimeDistributed(BatchNormalization(), name='frcnn_pose_bn1')(x, training=train_bn)(x)
    x = keras.layers.Activation(tf.nn.relu)(x)

    x = keras.layers.TimeDistributed(Conv2D(1024, (pool_size, pool_size), padding="valid"), name="frcnn_pose_conv2")(x)
    x = keras.layers.TimeDistributed(BatchNormalization(), name='frcnn_pose_bn2')(x, training=train_bn)(x)
    x = keras.layers.Activation(tf.nn.relu)(x)


    x = keras.layers.TimeDistributed(Conv2D(384, (1, 1)), name="frcnn_pose_conv3")(x)
    x = keras.layers.TimeDistributed(BatchNormalization(), name='frcnn_pose_bn3')(x, training=train_bn)(x)
    x = keras.layers.Activation(tf.nn.relu)(x)
    shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze_pose")(x)


    shared = keras.layers.TimeDistributed(Dense(num_classes * 4, activation='linear'))(shared)

    # Pose head
    # [batch, boxes, num_classes * (rx, ry, rz, tz)]
    x = keras.layers.TimeDistributed(Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_pose_fc')(shared)
    # Reshape to [batch, boxes, num_classes, (rx, ry, rz, tz)]
    s = K.int_shape(x)
    frcnn_pose = keras.layers.Reshape((s[1], num_classes, 4), name="mrcnn_pose")(x)

    return keras.Model([base_layers, rpn], frcnn_pose)


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    # identity block time distributed
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis, name="fixed_batch_normalisation_2Da"),  name=bn_name_base + '2Da')(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis, name="fixed_batch_normalisation_2Rb"), name=bn_name_base + '2Rb')(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis, name="fixed_batch_normalisation_2Tc"), name=bn_name_base + '2Tc')(x)
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def classifier_layers(x, input_shape, trainable=False):
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)
    return x


def Classifier(image, base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    out_roi_pool = ROIPooling(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return keras.Model([image, input_rois], [out_class, out_regr],  name="classifier_model")




def MobileNet(input, roi_input,  num_anchors, classes_count, alpha = 1.0, depth_multiplier=1, num_priors=[4, 6, 6, 6, 4, 4]):
    x = _conv_block(input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block_classification(x, 64, alpha, depth_multiplier, block_id=1)
    x = _depthwise_conv_block_classification(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block_classification(x, 128, alpha, depth_multiplier, block_id=3)
    x = _depthwise_conv_block_classification(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block_classification(x, 256, alpha, depth_multiplier, block_id=5)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=10)
    conv4_3 = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=11)
    shared = light_head(conv4_3)

    return [shared, roi_input, num_anchors, Config.num_rois, classes_count]



