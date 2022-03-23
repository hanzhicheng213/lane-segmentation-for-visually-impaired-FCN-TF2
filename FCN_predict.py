import cv2
import keras.layers
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import os
import glob
import datetime

scal = 224
path = r".\test.jpeg"
real_mask = r"D:\python_file\PycharmProjects\learn_deeplearning\Blind-road-and-crosswalk-dataset-master\Blind road and crosswalk dataset\test_labels\99.png"
video_path = r".\mangdao.mp4"
def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def normal_img(input_images):
    input_images = tf.cast(input_images, tf.float32)
    input_images = input_images/127.5 - 1
    input_images = tf.expand_dims(input_images, axis=0)
    return input_images

def load_images(input_images_path):
    input_images = read_jpg(input_images_path)
    input_images = tf.image.resize(input_images, (224, 224))
    return normal_img(input_images)

model = tf.keras.models.load_model('FCN.h5')

def pred_image(path, real_mask,model):
    # real_mask = Image.open(real_mask)
    img = load_images(path)
    print(img.shape)
    pred_mask = model.predict(img)
    pred_mask = tf.argmax(pred_mask, axis=-1)  # 对每个像素点分类结果取最大值的索引，放到最后一个维度上，图像变成（224*224）
    pred_mask = pred_mask[..., tf.newaxis]  # 拓展维度，（224，224）-->(224,224,1)
    pred_mask = tf.squeeze(pred_mask)
    print(pred_mask)

    image = read_jpg(path)
    image = tf.image.resize(image, [scal, scal])
    image = image / 127.5 - 1

    plt.imshow(image.numpy())
    plt.show()
    plt.imshow(pred_mask)
    plt.show()
    # plt.imshow(real_mask)
    # plt.show()

pred_image(path, real_mask, model)
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 3, 1)
# plt.imshow(image)
# plt.figure(1, 3, 2)
# plt.imshow(real_mask)
# plt.figure(1, 3, 3)
# plt.imshow(pred_mask)
# plt.show()