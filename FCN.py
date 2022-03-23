import keras.layers
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import os
import glob
import datetime

images_train = glob.glob(r".\Blind road and crosswalk dataset\train\*.jpg")
images_val = glob.glob(r".\Blind road and crosswalk dataset\val\*.jpg")
anno_train = glob.glob(r".\Blind road and crosswalk dataset\train_labels_new\*.png")
anno_val = glob.glob(r".\Blind road and crosswalk dataset\val_labels_new\*.png")


np.random.seed(2022)
index_train = np.random.permutation(len(images_train))
index_val = np.random.permutation(len(images_val))
images_train = np.array(images_train)[index_train]
images_val = np.array(images_val)[index_val]
anno_train = np.array(anno_train)[index_train]
anno_val = np.array(anno_val)[index_val]

dataset_train = tf.data.Dataset.from_tensor_slices((images_train, anno_train))
dataset_val = tf.data.Dataset.from_tensor_slices((images_val, anno_val))
train_count = len(images_train)
val_count = len(images_val)
# print(train_count)
# print(val_count)

def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def read_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    # img = img.numpy()
    # img[img == 14] = 1
    # img[img == 113] = 2
    # img = tf.convert_to_tensor(img)
    return img

def normal_img(input_images, input_anno):
    input_images = tf.cast(input_images, tf.float32)
    input_images = input_images/127.5 - 1
    return input_images, input_anno

def load_images(input_images_path, input_anno_path):
    input_images = read_jpg(input_images_path)
    input_anno = read_png(input_anno_path)
    input_images = tf.image.resize(input_images, (224, 224))
    input_anno = tf.image.resize(input_anno, (224, 224))
    return normal_img(input_images, input_anno)

dataset_train = dataset_train.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
dataset_val = dataset_val.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)

BATCH_SIZE = 8
dataset_train = dataset_train.repeat().shuffle(100).batch(BATCH_SIZE)
dataset_val = dataset_val.batch(BATCH_SIZE)
print(dataset_train)

# for img, anno in dataset_train.take(1):
#     plt.subplot(1, 2, 1)
#     plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
#     plt.subplot(1, 2, 2)
#     plt.imshow(tf.keras.preprocessing.image.array_to_img(anno[0]))
#     plt.show()

conv_base = tf.keras.applications.VGG16(weights='imagenet',
                                        input_shape=(224, 224, 3),
                                        include_top=False)
print(conv_base.summary())

# 创建一个子模型，得到中间层输出
# sub_model = tf.keras.models.Model(inputs=conv_base.input,
#                                   outputs=conv_base.get_layer('block5_conv3').output)
# print(sub_model.summary())
layers_name = [
    "block5_conv3",
    "block4_conv3",
    "block3_conv3",
    "block5_pool"
]
layers_output = [conv_base.get_layer(layer_name).output for layer_name in layers_name]
multiple_output_model = tf.keras.models.Model(inputs=conv_base.input,
                                              outputs=layers_output)
multiple_output_model.trainable = False

inputs = tf.keras.layers.Input(shape=(224, 224, 3)) # 构建网络第一层 输入层
out1, out2, out3, out = multiple_output_model(inputs)
# print(out.shape)
# print(out1.shape)
# print(out2.shape)
# print(out3.shape)

# 转置卷积
x1 = keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(out) # (卷积核个数， 卷积核大小， 步数决定扩大的倍数， padding方式, relu)
# print(x1.shape)
# 增加一层1卷积提取特征, 默认stride==1
x1 = keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x1)
# print(x1.shape)
# 特征图相加
x2 = tf.add(x1, out1)
# print(x2.shape)
x2 = keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(x2)
# print(x2.shape)
x2 = keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x2)
# print(x2.shape)
x3 = tf.add(x2, out2)
# print(x3.shape)
x3 = keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x3)
# print(x3.shape)
x3 = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x3)
# print(x3.shape)
x4 = tf.add(x3, out3)
# print(x4.shape)
# 放大一倍
x5 = keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x4)
# print(x5.shape)
x5 = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x5)
# print(x5.shape)
prediction = keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='softmax')(x5)
# print(prediction.shape)
model = tf.keras.models.Model(
    inputs=inputs,
    outputs=prediction
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
log_dir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))    # 保存路径
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)    # （保存路径， 每次epoch记录一次）
history = model.fit(
    dataset_train,
    epochs=50,
    steps_per_epoch=train_count//BATCH_SIZE,
    validation_data=dataset_val,
    validation_steps=val_count//BATCH_SIZE,
    callbacks=[tensorboard_callback]
)
model.save('FCN.h5')