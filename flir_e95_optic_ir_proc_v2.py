import collections
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from shutil import rmtree
from zipfile import ZipFile

import cv2
import face_alignment
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import get_face_features, face_add_eye_canthus, plot_image

ZIP_FP = Path("datasets/cold_shower_tympanic_98.7.zip")
TARGET_DIR = Path("datasets/E95_dataset_rgb_thermal_raw_98_7")
IR_IMG_SIZE = (348, 464)
TMP_DIR = Path(tempfile.mkdtemp("tmp"))
MARKER_THICKNESS = 3


def create_data_dir():
    if os.path.exists(TARGET_DIR):
        rmtree(TARGET_DIR)
    os.mkdir(str(TARGET_DIR))
    os.mkdir(str(TARGET_DIR / "RAW_IMAGES"))
    os.mkdir(str(TARGET_DIR / "RGB_IMAGES"))
    Path(TARGET_DIR / ".gitkeep").touch()
    try:
        with ZipFile(ZIP_FP, 'r') as fzip:
            fzip.extractall(TMP_DIR / ZIP_FP.stem)
        print('Extracting {} file(s) from the zip...'.format(len(os.listdir(TMP_DIR / ZIP_FP.stem))))
        for file in os.listdir(TMP_DIR / ZIP_FP.stem):
            if file.endswith(".csv"):
                temp_file = pd.read_csv(
                    TMP_DIR / ZIP_FP.stem / file,
                    header=None, sep=";")
                print("Number of thermal images - {}".format(temp_file.shape[0]))
                print("Generating the raw IR images...")
                for i in tqdm(range(len(temp_file))):
                    img_arr = temp_file.iloc[i].values[1:-1]
                    img_arr = np.reshape(img_arr, IR_IMG_SIZE)
                    img_arr = img_arr.astype('float64')
                    cv2.imwrite(
                        str(str(TARGET_DIR / "RAW_IMAGES" / "RAW_{}_src1.jpg".format(i + 1))),
                        img_arr)
            elif file.endswith('.jpg') and "optical" in file:
                shutil.copy(
                    TMP_DIR / ZIP_FP.stem / file,
                    TARGET_DIR / "RGB_IMAGES")
            else:
                pass
    except Exception as ex:
        print("Failed cause of exception - ", ex)
        print(traceback.format_exc())
    finally:
        print("Removing the tmp directory - ", TMP_DIR.name)
        rmtree(TMP_DIR, ignore_errors=True)


def ir_image_processing(ir_img, temp_csv, marker_thickness):
    in_img = ir_img.copy()
    kernel = np.array(
        [[0, 0, -2, 0, 0],
         [0, -2, -2, -2, 0],
         [-2, -2, 25, -2, -2],
         [0, -2, -2, -2, 0],
         [0, 0, -2, 0, 0]])
    # in_img = cv2.GaussianBlur(in_img, (15, 15), 0)
    in_img = cv2.filter2D(in_img, -1, kernel)
    faces_location_list, faces_landmark_list = get_face_features(in_img)
    if faces_location_list is None:
        print('WARN - No faces found in the image')
        return in_img
    for face, features in zip(faces_location_list, faces_landmark_list):
        in_img = cv2.rectangle(
            in_img,
            (face[3], face[0]),  # left top
            (face[1], face[2]),  # right bottom
            color=(0, 255, 0),
            thickness=marker_thickness)
        left_eye, right_eye = features['left_eye'], features['right_eye']
        eye_canthus_left = [x for x in left_eye if x[0] == max([x[0] for x in left_eye])][0]
        eye_canthus_right = [x for x in right_eye if x[0] == min([x[0] for x in right_eye])][0]
        in_img = get_temperature_n_mask(
            in_img, temp_csv, marker_thickness, eye_canthus_left, eye_canthus_right)
    return in_img


def ir_image_processing_v2(ir_img, temp_csv, marker_thickness):
    in_img = ir_img.copy()
    in_img = cv2.GaussianBlur(in_img, (15, 15), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(in_img)
    out_image = get_temperature_n_mask(in_img, temp_csv, marker_thickness, maxLoc)
    return out_image


def get_temp_in_celcius(temp):
    return np.round((temp * (9 / 5)) + 32, 1)


def get_mask_box(point, num_pix_above, num_pix_below=None):
    num_pix_below = num_pix_below if num_pix_below is not None else num_pix_above
    top_left = (point[0] - num_pix_above, point[1] - num_pix_above)
    bottom_right = (point[0] + num_pix_below, point[1] + num_pix_below)
    return top_left, bottom_right


def get_temperature_n_mask(img, temp_csv, marker_thickness, *points):
    for counter, point in enumerate(points):
        img = cv2.circle(img, point, 3, (0, 255, 255), marker_thickness)
        top_left, bottom_right = get_mask_box(point, num_pix_above=10)
        img = cv2.rectangle(
            img,
            top_left, bottom_right,
            (0, 255, 255), marker_thickness)
        temp_selected = temp_csv.values[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        min_temp, max_temp, avg_temp = \
            get_temp_in_celcius(np.min(temp_selected)), \
            get_temp_in_celcius(np.max(temp_selected)), \
            get_temp_in_celcius(np.mean(temp_selected))
        text_str = "Pt{} {}F | {}F | {}F".format(counter + 1, min_temp, avg_temp, max_temp)
        img = cv2.putText(
            img, text_str,
            (20, 20 + 30 * counter), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            0, 2, cv2.LINE_AA)
    return img


def run_face_detection(ir_dir_fp, rgb_dir_fp):
    for counter, fp in enumerate(zip(ir_dir_fp.iterdir(), rgb_dir_fp.iterdir())):
        ir_fp, rgb_fp = fp[0], fp[1]
        print("Reading IR - {} | RGB - {}".format(ir_fp.stem, rgb_fp.stem))
        ir_img = cv2.imread(str(ir_fp), cv2.IMREAD_GRAYSCALE)
        rgb_img = cv2.imread(str(rgb_fp), cv2.IMREAD_GRAYSCALE)
        # rgb_img_resize = cv2.resize(
        #
        # )
        # temp_csv = pd.read_csv(
        #     str(TARGET_DIR / "TEMP_ARRAYS" / "{}.{}".format(ir_fp.stem, "csv")),
        #     header=None, sep=';')
        out_img = face_add_eye_canthus(rgb_img)
        print('RGB shape - ', out_img.shape)
        print('IR shape - ', ir_img.shape)
        plot_image(out_img)
        if counter == 5:
            break


def run():
    create_data_dir()
    # run_face_detection(
    #     TARGET_DIR / "RAW_IMAGES",
    #     TARGET_DIR / "RGB_IMAGES")


if __name__ == '__main__':
    run()
