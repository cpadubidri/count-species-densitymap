import keras
from keras.layers import LeakyReLU
from Datagen import root_mean_squared_error
import segmentation_models as sm
import os


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    p = keras.layers.BatchNormalization(trainable=True)(p)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    us = keras.layers.BatchNormalization()(us)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(c)
    c = keras.layers.BatchNormalization(trainable=True)(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=LeakyReLU(alpha=0.1),kernel_initializer=keras.initializers.he_uniform(seed=None))(c)
    return c


def UNet(image_size=256):
    f = [16, 32, 64, 128, 256, 512]
    inputs = keras.layers.Input((image_size, image_size, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])  # 128 -> 64
    c2, p2 = down_block(p1, f[1])  # 64 -> 32
    c3, p3 = down_block(p2, f[2])  # 32 -> 16
    c4, p4 = down_block(p3, f[3])  # 16->8
    c5, p5 = down_block(p4, f[4])  # 16->8

    bn = bottleneck(p5, f[5])

    u1 = up_block(bn, c5, f[4])  # 8 -> 16
    u2 = up_block(u1, c4, f[3])  # 16 -> 32
    u3 = up_block(u2, c3, f[2])  # 32 -> 64
    u4 = up_block(u3, c2, f[1])  # 64 -> 128
    u5 = up_block(u4, c1, f[0])  # 32 -> 64

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u5)
    outputs = keras.layers.Conv2D(1, (11, 11), padding="same", activation="sigmoid")(outputs)
    model = keras.models.Model(inputs, outputs)
    return model
# 0.00009

def basic_model(image_size=256):
    unet_model = UNet(image_size=image_size)
    optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
    metrics = ["mean_squared_error","mean_squared_logarithmic_error","mean_absolute_error","squared_hinge"]
    unet_model.compile(optimizer=optimizer, loss=root_mean_squared_error, metrics=metrics)
    # unet_model.summary()
    unet_model.load_weights('./Training_Logs/Pretrained/Elephant_Model_EOT.h5')
    return unet_model

def tl_model(image_size=256):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    BACKBONE = 'efficientnetb5'
    CLASSES = ['elephant']
    LR = 0.0001

    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    # create model
    unet_model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, encoder_weights='imagenet')

    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5),"mean_squared_error","mean_squared_logarithmic_error","mean_absolute_error"]
    unet_model.compile(optim, root_mean_squared_error, metrics)
    unet_model.summary()
    return unet_model



if __name__ == '__main__':
    model = basic_model()