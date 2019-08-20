import json
import os

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Bidirectional
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.models import Model

import cv2

#MEANS = np.array([165.9459838, 170.92398165, 170.94181033])
#STDS = np.array([57.29009752, 57.78994836, 57.70681251])
charset = "0123456789<abcdefghijklmnopqrstuvwxyz"
idx2char = {i: c for i, c in enumerate(charset)}
SIZE = (75,75)
CLASSES = 37

def build_model(trainable=False, weights='imagenet'):
    # init model
    inputs = Input(name='inputs', shape=(SIZE[0], SIZE[1], 3), dtype='float32')
    vgg = keras.applications.InceptionV3(weights=weights, include_top=False)
    vgg_tensor = vgg(inputs)
    flatten = Reshape(target_shape=(-1,), name='flatten')(vgg_tensor)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal', name='fc')(flatten)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_initializer='he_normal', name='fc2')(x)
    x = BatchNormalization()(x)
    output = Dense(CLASSES, activation='softmax', kernel_initializer='he_normal', name='output')(x)

    for layer in vgg.layers:
        layer.trainable = trainable

    model =  Model(inputs=[inputs], outputs=output)
    return model

# load model
ocr_graph = tf.Graph()
MODEL_OCR = None
with ocr_graph.as_default():
    session_ocr = tf.Session()
    with session_ocr.as_default():
        MODEL_OCR = build_model(trainable=True, weights=None)
        MODEL_OCR.load_weights('api/ocr/classify/model.h5')


def decode_predict(predict):
    indices = predict.argmax(axis=1)
    predicted_chars = [idx2char[i] for i in indices]
    return predicted_chars

def predict_batch_images(images):
    images = images.astype(np.float32)
    images /= 255
    images = images * 2 - 1
    # for channelidx in range(3):
    #     images[:,:,:,channelidx] = (images[:,:,:,channelidx] - MEANS[channelidx]) / STDS[channelidx]

    with ocr_graph.as_default():
        with session_ocr.as_default():
            predict = MODEL_OCR.predict(images)
            predict = decode_predict(predict)
            return predict
