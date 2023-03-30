import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

####################
### define Adain ###
####################
def expand_moments_dim(moment):
    return tf.reshape(moment, [-1, 1, 1, tf.shape(moment)[-1]])

def adain(content_feature, style_feature, eps=1e-5):
    content_mean, content_var = tf.nn.moments(content_feature, axes=[1, 2])
    style_mean, style_var = tf.nn.moments(style_feature, axes=[1, 2])

    content_std = tf.sqrt(content_var)
    style_std = tf.sqrt(style_var)

    content_mean = expand_moments_dim(content_mean)
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # content_mean = tf.broadcast_to(content_mean, tf.shape(content_feature))

    content_std = expand_moments_dim(content_std) + eps
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # content_std = tf.broadcast_to(content_std, tf.shape(content_feature))

    style_mean = expand_moments_dim(style_mean)
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # style_mean = tf.broadcast_to(style_mean, tf.shape(content_feature))

    style_std = expand_moments_dim(style_std) + eps
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # style_std = tf.broadcast_to(style_std, tf.shape(content_feature))

    normalized_content = tf.divide(content_feature - content_mean, content_std)
    return tf.multiply(normalized_content, style_std) + style_mean



####################
### Define model ###
####################
class YAE(tf.keras.Model):
    """Conditional variational autoencoder."""
    def __init__(self, opt, num_character):
        super(YAE, self).__init__()
        self.channels = opt.channels
        self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.bias_initializer_implicit = tf.constant_initializer(-5.0)
        self.img_size = opt.img_size
        self.conv_filters = 64
        self.style_size = self.conv_filters*8
        self.content_size = num_character 
        self.ksize = 2

        self.encoder = self.encoder_model()
        self.decoder = self.decoder_model()

    #####################
    ### Encoder model ###
    #####################
    def conv2d(self, inp, fil, k=3, st=1, dila=1, apply_norm=True, apply_activation=True):
        y = tf.keras.layers.Conv2D(filters=fil
                                 , kernel_size=k
                                 , strides=st
                                 , padding='same'
                                 , kernel_initializer= self.initializer
                                 , dilation_rate=(dila, dila)
                                 , kernel_regularizer=tf.keras.regularizers.l2(0.00001))(inp)
        if apply_norm:
            y = tf.keras.layers.BatchNormalization()(y)

        if apply_activation:
            y = tf.keras.layers.ReLU()(y)

        return y

    def ResBlock(self, inp, fil, stride=1, p=False):
        out = self.conv2d(inp, fil, 3, stride)
        out = self.conv2d(out, fil, 3, apply_activation=False)

        if p:
            inp = tf.keras.layers.BatchNormalization()(inp)
            inp = self.conv2d(inp, fil, 1, stride, apply_activation=False)

        out = tf.keras.layers.add([inp, out])

        return out

    def encoder_model(self):
        inputs = tf.keras.layers.Input(shape=[self.img_size, self.img_size, self.channels])

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=7
                                 , strides=2, padding='same')(inputs)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = self.ResBlock(x, 64)
        x = self.ResBlock(x, 64)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = self.ResBlock(x, 128, stride=2, p=True)
        x = self.ResBlock(x, 128)

        x = self.ResBlock(x, 256, stride=2, p=True)
        x = self.ResBlock(x, 256)

        x = self.ResBlock(x, 512, stride=2, p=True)
        x = self.ResBlock(x, 512)



        style = tf.keras.layers.Conv2D(filters=self.style_size
                                     , kernel_size=self.ksize
                                     , strides=2
                                     , padding='same'
                                     , kernel_regularizer=tf.keras.regularizers.l2(0.00001)
                                     , bias_initializer=self.bias_initializer_implicit)(x)
        output_style = tf.keras.layers.GlobalAveragePooling2D()(style)
        output_style = tf.keras.activations.sigmoid(output_style)
        output_style = tf.keras.layers.Dropout(0.2)(output_style)

        content1 = tf.keras.layers.GlobalAveragePooling2D()(x)
        # content1 = tf.keras.layers.Dense(1024, activation='relu')(content1)
        output_content1 = tf.keras.layers.Dense(self.content_size, activation='softmax')(content1)

        return tf.keras.Model(inputs=inputs
                            , outputs=[output_style
                                     , output_content1])

    #####################
    ### Decoder model ###
    #####################
    def conv2d_t(self, layer_input, filters, stride=2, apply_norm=True, apply_activation=True, apply_dropout=False):
        y = tf.keras.layers.Conv2DTranspose(filters=filters
                                          , kernel_size=self.ksize
                                          , strides=stride
                                          , padding='same'
                                          , kernel_initializer= self.initializer #)(layer_input)
                                          , kernel_regularizer=tf.keras.regularizers.l2(0.00001))(layer_input)
        if apply_norm:
            # y = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(y)
            y = tfa.layers.InstanceNormalization(axis = -1, center = False , scale = False)(y)

        if apply_dropout:
            y = tf.keras.layers.Dropout(0.5)(y)

        if apply_activation:
            y = tf.keras.layers.ReLU()(y)
        else :
            y = tf.keras.layers.Activation('sigmoid')(y)

        return y

    def decoder_model(self):
        input_img = tf.keras.layers.Input(shape=[self.style_size, ]
                                        , dtype=tf.float32)

        input_label = tf.keras.layers.Input(shape=[1], dtype=tf.float32)
        label_e = tf.keras.layers.Embedding(self.content_size , 512)(input_label)
        label_e = tf.keras.layers.Reshape((512, ))(label_e)

        l1 = tf.keras.layers.Dense(units=2*2*self.conv_filters*16)(label_e)
        l1 = tf.keras.layers.Activation('relu')(l1)
        l1_in = tf.keras.layers.Reshape((2, 2, self.conv_filters*16))(l1)

        l2 = tf.keras.layers.Dense(units=4*4*self.conv_filters*16)(label_e)
        l2 = tf.keras.layers.Activation('relu')(l2)
        l2_in = tf.keras.layers.Reshape((4, 4, self.conv_filters*16))(l2)

        l3 = tf.keras.layers.Dense(units=8*8*self.conv_filters*8)(label_e)
        l3 = tf.keras.layers.Activation('relu')(l3)
        l3_in = tf.keras.layers.Reshape((8, 8, self.conv_filters*8))(l3)

        l4 = tf.keras.layers.Dense(units=16*16*self.conv_filters*8)(label_e)
        l4 = tf.keras.layers.Activation('relu')(l4)
        l4_in = tf.keras.layers.Reshape((16, 16, self.conv_filters*8))(l4)

        di = tf.keras.layers.Reshape(target_shape=(1, 1, self.style_size))(input_img)
                                                    # (1, 1, )
        d1 = self.conv2d_t(di, self.conv_filters*16)#, apply_dropout=True)   # (2, 2, )
        dl1 = adain(l1_in, d1)

        d2 = self.conv2d_t(dl1, self.conv_filters*16)#, apply_dropout=True)   # (4, 4, )
        dl2 = adain(l2_in, d2)

        d3 = self.conv2d_t(dl2, self.conv_filters*8)#, apply_dropout=True)   # (8, 8, )
        dl3 = adain(l3_in, d3)

        d4 = self.conv2d_t(dl3, self.conv_filters*8)   # (16, 16, )
        dl4 = adain(l4_in, d4)

        d5 = self.conv2d_t(dl4, self.conv_filters*4)   # (32, 32, )
        # dl5 = adain(l5_in, d5)

        d = self.conv2d_t(d5, self.conv_filters*4)   # (64, 64, )
        d = self.conv2d_t(d, self.conv_filters*2)   # (128, 128, )
        d = self.conv2d_t(d, self.conv_filters*1)   # (256, 256, )

        output = self.conv2d_t(d, self.channels, 1, apply_activation=False)

        # dr2 = self.conv2d(d, self.conv_filters, 3, 1, 2)
        # dr3 = self.conv2d(d, self.conv_filters, 3, 1, 3)
        # dr4 = self.conv2d(d, self.conv_filters, 3, 1, 4)
        # dr5 = self.conv2d(d, self.conv_filters, 3, 1, 5)

        # cat = tf.concat([dr2, dr3, dr4, dr5], 3)

        # output = self.conv2d_t(cat, self.channels, 1, apply_activation=False)

        return tf.keras.Model(inputs=[input_img, input_label], outputs=output)
