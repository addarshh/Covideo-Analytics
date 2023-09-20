import keras
import tensorflow as tf
import numpy as np
import PIL
import cv2
import csv
from ast import literal_eval
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import backend as K, Model
from keras import losses
from keras.layers import Dense
from sklearn.model_selection import train_test_split
#from fcn_8 import make_vgg16fcn8s_model

TRAIN_DATA = './preprocess/'
HUBER_DELTA = 0.5
DATASET_SIZE = 24000


def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


def create_training_data():
    training_data = []
    # zeros = []
    # for i in range(0, 50176):
    #     zeros.append(0)
    base_class = np.zeros([256, 256,1])

    with open("top_1000_face_and_eye_data.csv", 'r') as file:
        csv_file = csv.DictReader(file)
        i = 0
        for row in csv_file:
            if i >= DATASET_SIZE:
                break

            eye_pts_all = literal_eval(row["eye_points"])


            if (len(eye_pts_all) != 12):
                continue

            eye_img_class = base_class.copy()

            eye_pts = eye_pts_all

            # full_classes = zeros.copy()
            # for (x,y)in eye_pts_all:
            #     full_classes[x]=1
            #     full_classes[y] = 1
            # eye_pts = full_classes
            # eye_pts=[]
            # eye_pts_L = eye_pts_all[:6]
            # eye_pts_R = eye_pts_all[6:12]
            #
            # sumx = 0
            # sumy = 0
            # for (x,y) in eye_pts_L:
            #     sumx += x
            #     sumy += y
            # eye_pts.append([int(sumx/len(eye_pts_L)),int(sumy/len(eye_pts_L))])
            # sumx = 0
            # sumy = 0
            # for (x,y) in eye_pts_R:
            #     sumx += x
            #     sumy += y
            # eye_pts.append([int(sumx/len(eye_pts_R)),int(sumy/len(eye_pts_R))])

            img_array = cv2.imread((TRAIN_DATA + row["name"]), cv2.IMREAD_GRAYSCALE)
            training_data.append([img_array, eye_pts])
            i += 1
    return training_data


def train():


    train_data = create_training_data()

    X = []
    y = []

    for features, label in train_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, 224, 224, 1)
    X = X / 255.0
    y = np.array(y).reshape(-1, 24)
    y=y/224

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = keras.applications.mobilenet_v2.MobileNetV2((224, 224, 1), alpha=1.0, include_top=False, weights=None, pooling='max')
    last_layer = model.layers[-1].output
    x = Dense(24, name='fc_14')(last_layer)
    model = Model(model.input,output=x)
    #model = make_vgg16fcn8s_model()


    model.compile(loss=smoothL1, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=25, epochs=200, shuffle=True, verbose=1, validation_data=(X_test, y_test))

    model.save("./face_landmark_mobilenet.h5")

    print('done')

if __name__ == '__main__':
    train()
