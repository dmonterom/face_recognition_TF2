import tensorflow as tf
import os
from arcface import Arcfacelayer

bn_axis = -1
initializer = 'glorot_normal'


class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape),
                                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.keras.backend.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = tf.keras.backend.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = tf.keras.backend.stack(group_shape)
        inputs = tf.keras.backend.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = tf.keras.backend.mean(
            inputs, axis=group_reduction_axes, keepdims=True)
        variance = tf.keras.backend.var(
            inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / \
            (tf.keras.backend.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = tf.keras.backend.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = tf.keras.backend.reshape(
                self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = tf.keras.backend.reshape(
                self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = tf.keras.backend.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


tf.keras.utils.get_custom_objects().update(
    {'GroupNormalization': GroupNormalization})


def residual_unit_v3(input, num_filter, stride, dim_match, name):
    x = GroupNormalization(axis=bn_axis,
                           scale=True,
                           epsilon=2e-5,
                           #    beta_regularizer=tf.keras.regularizers.l2(
                           #        l=5e-4),
                           gamma_regularizer=tf.keras.regularizers.l2(
                               l=5e-4),
                           name=name + '_bn1')(input)
    x = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name=name + '_conv1_pad')(x)
    x = tf.keras.layers.Conv2D(num_filter, (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(
                                   l=5e-4),
                               name=name + '_conv1')(x)
    x = GroupNormalization(axis=bn_axis,
                           scale=True,
                           epsilon=2e-5,
                           #    beta_regularizer=tf.keras.regularizers.l2(
                           #        l=5e-4),
                           gamma_regularizer=tf.keras.regularizers.l2(
                               l=5e-4),
                           name=name + '_bn2')(x)
    x = tf.keras.layers.PReLU(name=name + '_relu1',
                              alpha_regularizer=tf.keras.regularizers.l2(
                                  l=5e-4))(x)
    x = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name=name + '_conv2_pad')(x)
    x = tf.keras.layers.Conv2D(num_filter, (3, 3),
                               strides=stride,
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(
                                   l=5e-4),
                               name=name + '_conv2')(x)
    x = GroupNormalization(axis=bn_axis,
                           scale=True,
                           epsilon=2e-5,
                           #    beta_regularizer=tf.keras.regularizers.l2(
                           #        l=5e-4),
                           gamma_regularizer=tf.keras.regularizers.l2(
                               l=5e-4),
                           name=name + '_bn3')(x)
    if (dim_match):
        shortcut = input
    else:
        shortcut = tf.keras.layers.Conv2D(num_filter, (1, 1),
                                          strides=stride,
                                          padding='valid',
                                          kernel_initializer=initializer,
                                          use_bias=False,
                                          kernel_regularizer=tf.keras.regularizers.l2(
                                              l=5e-4),
                                          name=name + '_conv1sc')(input)
        shortcut = GroupNormalization(axis=bn_axis,
                                      scale=True,
                                      epsilon=2e-5,
                                      #   beta_regularizer=tf.keras.regularizers.l2(
                                      #       l=5e-4),
                                      gamma_regularizer=tf.keras.regularizers.l2(
                                          l=5e-4),
                                      name=name + '_sc')(shortcut)
    return x + shortcut


def get_fc1(input):
    x = GroupNormalization(axis=bn_axis,
                           scale=True,
                           epsilon=2e-5,
                           #    beta_regularizer=tf.keras.regularizers.l2(
                           #        l=5e-4),
                           gamma_regularizer=tf.keras.regularizers.l2(
                               l=5e-4),
                           name='bn1')(input)
    x = tf.keras.layers.Dropout(0.4)(x)
    resnet_shape = input.shape
    x = tf.keras.layers.Reshape(
        [resnet_shape[1] * resnet_shape[2] * resnet_shape[3]], name='reshapelayer')(x)
    x = tf.keras.layers.Dense(512,
                              name='E_DenseLayer', kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l2(
                                  l=5e-4),
                              bias_regularizer=tf.keras.regularizers.l2(
                                  l=5e-4))(x)
    x = GroupNormalization(axis=-1,
                           scale=False,
                           epsilon=2e-5,
                           #    beta_regularizer=tf.keras.regularizers.l2(
                           #        l=5e-4),
                           name='fc1')(x)
    return x


def ResNet50():

    input_shape = [112, 112, 3]
    filter_list = [64, 64, 128, 256, 512]
    units = [3, 4, 14, 3]
    num_stages = 4

    img_input = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name='conv0_pad')(img_input)
    x = tf.keras.layers.Conv2D(64, (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(
                                   l=5e-4),
                               name='conv0')(x)
    x = GroupNormalization(axis=bn_axis,
                           scale=True,
                           epsilon=2e-5,
                           #    beta_regularizer=tf.keras.regularizers.l2(
                           #        l=5e-4),
                           gamma_regularizer=tf.keras.regularizers.l2(
                               l=5e-4),
                           name='bn0')(x)
    # x = tf.keras.layers.Activation('prelu')(x)
    x = tf.keras.layers.PReLU(
        name='prelu0',
        alpha_regularizer=tf.keras.regularizers.l2(
            l=5e-4))(x)

    for i in range(num_stages):
        x = residual_unit_v3(x, filter_list[i + 1], (2, 2), False,
                             name='stage%d_unit%d' % (i + 1, 1))
        for j in range(units[i] - 1):
            x = residual_unit_v3(x, filter_list[i + 1], (1, 1),
                                 True, name='stage%d_unit%d' % (i + 1, j + 2))

    x = get_fc1(x)

    # Create model.
    model = tf.keras.models.Model(img_input, x, name='resnet50')
    model.trainable = True
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
        # if ('conv0' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('bn0' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('prelu0' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage1' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage2' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage3' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage4' in model.layers[i].name):
        #     model.layers[i].trainable = False

    return model


class train_model(tf.keras.Model):
    def __init__(self):
        super(train_model, self).__init__()
        self.resnet = ResNet50()
        self.arcface = Arcfacelayer()

    def call(self, x, y):
        x = self.resnet(x)
        return self.arcface(x, y)
