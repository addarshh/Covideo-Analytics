from multiprocessing.dummy import Pool as ThreadPool
# import numpy as np
import cv2
from os import walk
# import pandas as pd
import dlib
import imutils
from imutils import face_utils
import hashlib
import math
import csv
import random


INPUT_PATH = "./IR_like_1000_VGGFace_data/"
OUTPUT_PATH = "./preprocess/"
THREAD_COUNT = 20
DATASET_SIZE = 30000 #!!will only take ~75% of this dataset size
IMG_SIZE = 224

def rect_to_bb(rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        # return a tuple of (x, y, w, h)
        return (x, y, w, h)

def get_eyes(shape, dtype="int"):
        eye_list=[]
        for i in range(36,48):
                eye_list.append([shape.part(i).x, shape.part(i).y])
        return  eye_list


def convert_to_semiIR(f):
    for img in f:
            if img =="":
                continue
            imin = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
            imin = cv2.resize(imin, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)
            heat = cv2.applyColorMap(imin, cv2.COLORMAP_HSV)
            gray = cv2.cvtColor(heat, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str((OUTPUT_PATH+hashlib.md5(str(img).encode('utf-8')).hexdigest()+".jpg")),gray)
    return 1

def get_bounding_boxes(f):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    data_to_save = {"name":[],"face_location":[],"eye_points":[]}

    for img in f:
        if img =="":
            continue
        imin = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(imin, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            eye_list = get_eyes(shape)

            data_to_save["name"].append(str((hashlib.md5(str(img).encode('utf-8')).hexdigest()+".jpg")))
            data_to_save["face_location"].append([x,y,w,h])
            data_to_save["eye_points"].append(eye_list)



    return data_to_save


def build_data():

    f = []
    for dirpath, dirnames, filenames in walk(INPUT_PATH):
        f.extend([dirpath+'/'+file for file in filenames])

    work_list = []

    f = random.choices(f, k=DATASET_SIZE)

    work_size = math.floor(len(f)/THREAD_COUNT)

    for i in range (0,THREAD_COUNT):
        work_list.append(f[(i*work_size):((i+1)*work_size)])


    # for i in range (0,math.floor(len(f)/20) ):
    # #for i in range(0, math.floor( 10/ 2)):
    #
    #     # work_array = []
    #     # # for j in range(i*20,(i+1)*20):
    #     # # #for j in range(i * 2, (i + 1) * 2):
    #     # #
    #     # #     work_array.append(f[j])
    #     # work_array.append(f[(i*20):((i+1)*20)])
    #     work_list.append(f[(i*20):((i+1)*20)])


    pool = ThreadPool(THREAD_COUNT)
    results = pool.map(convert_to_semiIR, work_list)
    results = pool.map(get_bounding_boxes, work_list)


    data_to_save = {"name": [], "face_location": [], "eye_points": []}
    for r in results:
        data_to_save["name"].extend(r["name"])
        data_to_save["face_location"].extend(r["face_location"])
        data_to_save["eye_points"].extend(r["eye_points"])

    with open('top_1000_face_and_eye_data.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data_to_save.keys())
        writer.writerows(zip(*data_to_save.values()))

if __name__ == '__main__':
    build_data()
