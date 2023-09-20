from os import walk

import keras
import tensorflow as tf
import numpy as np
import cv2
import PIL
from keras import backend as K


HUBER_DELTA = 0.5
IN_DIR = "IR_like_1000_VGGFace_data/"
OUT_DIR = "test_model_out/"

def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)

model = keras.models.load_model('face_landmark_mobilenet.h5',compile=False)
model.compile(loss=smoothL1, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])





for dirpath, dirnames, filenames in walk(IN_DIR):
    for file in filenames:
        if not "jpg" in file:
            continue
        img = cv2.imread(IN_DIR+file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        test = np.array(img).reshape(1, 224, 224, 1)
        test = test / 255

        out = model.predict([test])

        out*=224

        it = iter(out[0])
        for x in it:
            #print (x, next(it))
            y = next(it)

            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

        cv2.imwrite(OUT_DIR+file, img)
# for (x, y) in out:
#     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)


print(out)