import os
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

class ModelUnet:
    def TL_unet_model(input_shape):
        # input: input_shape (height, width, channels) 
        # return model
        input_shape = input_shape
        base_VGG = VGG16(include_top = False, 
                    weights = "imagenet", 
                    input_shape = input_shape)

        # freezing all layers in VGG16 
        for layer in base_VGG.layers: 
            layer.trainable = False

        # the bridge (exclude the last maxpooling layer in VGG16) 
        bridge = base_VGG.get_layer("block5_conv3").output
        print(bridge.shape)

        # Decoder
        up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
        print(up1.shape)
        concat_1 = concatenate([up1, base_VGG.get_layer("block4_conv3").output], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_1)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
        print(up2.shape)
        concat_2 = concatenate([up2, base_VGG.get_layer("block3_conv3").output], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat_2)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
        print(up3.shape)
        concat_3 = concatenate([up3, base_VGG.get_layer("block2_conv2").output], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat_3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
        print(up4.shape)
        concat_4 = concatenate([up4, base_VGG.get_layer("block1_conv2").output], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat_4)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9) #2 classes wear and tool
        print(conv10.shape)

        model_ = Model(inputs=[base_VGG.input], outputs=[conv10])

        return model_ # returns keras model
    
    # Evaluation metrics: dice coefficient 
    def dice_coef(y_true, y_pred, smooth = 1.):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return - ModelUnet.dice_coef(y_true, y_pred)

    # Evaluation metrics: iou
    def iou(y_true, y_pred, smooth = 1.):
        intersection = K.sum(y_true * y_pred)
        sum_ = K.sum(y_true) + K.sum(y_pred)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return jac

    
