from flask import Blueprint
from flask import request

import warnings
warnings.filterwarnings('ignore')
from keras.utils import to_categorical
import segmentation_models as sm
import numpy as np
import cv2
import os
import keras
keras.backend.set_image_data_format('channels_last')
import keras.backend as K
import tensorflow as tf


unet = Blueprint('unet', __name__)
train_x, train_y = [], []
val_x, val_y = [], []


# 图像格式转换
@unet.route('/unet_1', methods=['POST'])
def transform():
    # image_dir = 'G:/remote_sensing/json100/dataset/3/image'
    base_dir = request.form.get('base_dir')
    print(base_dir)
    print(os.getcwd())
    val_split = request.form.get('val_split')
    # print(val_split)
    if not val_split:
        val_split = 0.2
    val_split = float(val_split)
    print(base_dir)
    image_dir = os.path.join(base_dir, 'image')
    label_dir = os.path.join(base_dir, 'label')
    image_num = len(os.listdir(image_dir))
    print('total image number', image_num)

    train_list, label_list = [], []
    for image in os.listdir(image_dir)[:]:
        train_list.append(os.path.join(image_dir, image))
    for label in os.listdir(label_dir)[:]:
        label_list.append(os.path.join(label_dir, label))

    index = np.arange(image_num)
    np.random.shuffle(index)
    train_split = int((1 - val_split) * image_num)
    print('train number', train_split)
    global train_x, train_y, val_x, val_y
    # print(index[:2])
    # print(index.shape, train_split)

    train_list, label_list = np.array(train_list), np.array(label_list)
    val_x, val_y = np.array(val_x), np.array(val_y)
    # print(index[:train_split])
    train_x, train_y = train_list[index[:train_split]], label_list[index[:train_split]]
    val_x, val_y = train_list[index[train_split:]], label_list[index[train_split:]]
    return 'transform has been done'


def get_batch(batch=4, dim=256, class_num=2):
    train_num = len(train_x)
    steps = train_num // batch
    index = np.arange(train_num)
    cur = 0
    while True:
        batch_x, batch_y = [], []
        if cur < steps:
            cur += 1
        else:
            cur = 1
            np.random.shuffle(index)
        for i in range((cur-1)*batch, cur*batch):
            im = cv2.imread(train_x[index[i]])[:dim, :dim, :]
            batch_x.append(im)
            lab = cv2.imread(train_y[index[i]], cv2.IMREAD_GRAYSCALE)[:dim, :dim]
            batch_y.append(lab)
        if np.max(batch_x) > 1:
            batch_x = np.array(batch_x) / 255
        batch_y = np.array(batch_y).reshape((-1, dim, dim, 1))
        batch_y_one_hot = to_categorical(batch_y, class_num)
        yield batch_x, batch_y_one_hot


def get_val(dim=256, class_num=2):
    val_x_list, val_y_list = [], []
    for i in range(len(val_x)):
        val_image = cv2.imread(val_x[i])[:dim, :dim, :]
        val_label = cv2.imread(val_y[i], cv2.IMREAD_GRAYSCALE)[:dim, :dim]
        val_x_list.append(val_image)
        val_y_list.append(val_label)
    # print('val_x_list', val_x_list)
    print(np.max(val_x_list))
    if np.max(val_x_list) > 1:
        val_x_list = np.array(val_x_list) / 255
    print(val_x_list.shape)
    val_y_list = np.array(val_y_list).reshape((-1, 256, 256, 1))
    val_y_list_onehot = to_categorical(val_y_list, class_num)
    return val_x_list, val_y_list_onehot


def recall(y_true, y_pred):
    true = K.argmax(y_true, axis=3)
    pred = K.argmax(y_pred, axis=3)
    true_positives = K.sum(K.round(K.clip(true * pred, 0, 1)))
    possible_positives = K.sum(true)
    recall = true_positives / possible_positives
    return recall


def precision(y_true, y_pred):
    true = K.argmax(y_true, axis=3)
    pred = K.argmax(y_pred, axis=3)
    true_positives = K.sum(K.round(K.clip(true * pred, 0, 1)))
    predicted_positives = K.sum(pred)
    if predicted_positives == 0:
        precision = 0
    else:
        precision = true_positives / predicted_positives
    return precision


@unet.route('/unet_2', methods=['POST'])
def train():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_dir = request.form.get('model_dir')
    batch = request.form.get('batch')
    class_num = request.form.get('class_num')
    period = request.form.get('period')
    epochs = request.form.get('epochs')
    dim = request.form.get('dim')
    BACKBONE = request.form.get('BACKBONE')
    print('model_dir', model_dir)
    if not batch:
        batch = 4
    else:
        batch = int(batch)
    if not class_num:
        class_num = 2
    else:
        class_num = int(class_num)
    if not period:
        period = 1
    else:
        period = int(period)
    if not epochs:
        epochs = 1000
    else:
        epochs = int(epochs)
    if not dim:
        dim = 256
    else:
        dim = int(dim)
    if not BACKBONE:
        BACKBONE = 'resnext50'

    print('class_num', class_num)
    print('BACKBONE', BACKBONE)
    # class_num = request.form.get(class_num)
    if class_num == 2:
        model = sm.Unet(BACKBONE, classes=class_num, activation='sigmoid', encoder_weights='imagenet')
        model.compile(
            keras.optimizers.Adam(lr=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', precision, recall]
        )
        ckpt = keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'Epoch_{epoch:03d}-{loss:.4f}-{acc:.4f}-{precision:.4f}-{recall:.4f}.h5'),
            period=period)
    else:
        model = sm.Unet(BACKBONE, classes=class_num, activation='softmax', encoder_weights='imagenet')
        model.compile(
            keras.optimizers.Adam(lr=3e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        ckpt = keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'Epoch_{epoch:03d}-{loss:.6f}-{acc:.6f}.h5'), period=period)
    train_num = len(train_x)
    model.fit_generator(
        get_batch(class_num=class_num, dim=dim),
        steps_per_epoch=train_num//batch*2,
        epochs=epochs,
        callbacks=[ckpt],
        class_weight='auto',
        validation_data=get_val(class_num=class_num, dim=dim)
    )
    return 'train done'


def run_train():
    transform('G:/remote_sensing/json100/dataset/3', val_split=0.2)
    print('trainsform done')
    train('G:/model/5', class_num=6)
    return 'test done'


# 1、实现UNet的房屋检测（5类别）模型模块化
if __name__ == '__main__':
    transform('G:/remote_sensing/json100/dataset/3', val_split=0.2)
    print('trainsform done')
    train('G:/model/5', class_num=6)




