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

from utils import get_face_features

ZIP_FP = Path("datasets/flir_e95_dataset1.zip")
TARGET_DIR = Path("datasets/E95_dataset")
RESULTS_DIR = Path("datasets/E95_dataset/RESULTS")
SAVE_RESULT = True
if SAVE_RESULT and not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir()
else:
    for f in RESULTS_DIR.iterdir():
        os.remove(str(f))
TMP_DIR = Path(tempfile.mkdtemp("tmp"))
MARKER_THICKNESS = 3

FaceAlignment = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D,
    device='cpu',
    flip_input=True)
PRED_TYPE = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': PRED_TYPE(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': PRED_TYPE(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': PRED_TYPE(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': PRED_TYPE(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': PRED_TYPE(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': PRED_TYPE(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': PRED_TYPE(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': PRED_TYPE(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': PRED_TYPE(slice(60, 68), (0.596, 0.875, 0.541, 0.4))}


def create_data_dir():
    if os.path.exists(TARGET_DIR):
        rmtree(TARGET_DIR)
    os.mkdir(str(TARGET_DIR))
    os.mkdir(str(TARGET_DIR / "IR_IMAGES"))
    os.mkdir(str(TARGET_DIR / "TEMP_ARRAYS"))
    Path(TARGET_DIR / ".gitkeep").touch()
    try:
        with ZipFile(ZIP_FP, 'r') as fzip:
            fzip.extractall(TMP_DIR / ZIP_FP.stem)
        print('Extracting {} file(s) from the zip...'.format(len(os.listdir(TMP_DIR / ZIP_FP.stem))))
        for file in os.listdir(TMP_DIR / ZIP_FP.stem):
            if file.endswith(".csv"):
                shutil.copy(
                    TMP_DIR / ZIP_FP.stem / file,
                    TARGET_DIR / "TEMP_ARRAYS")
            elif file.endswith('.jpg'):
                shutil.copy(
                    TMP_DIR / ZIP_FP.stem / file,
                    TARGET_DIR / "IR_IMAGES")
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


def ir_image_processing_v3(ir_img, temp_csv, marker_thickness):
    in_img = ir_img.copy()
    preds = FaceAlignment.get_landmarks(in_img)
    if preds is not None:
        preds = preds[-1]
        preds = preds.astype('int32')
        left_eye, right_eye = preds[pred_types['eye1'].slice], preds[pred_types['eye2'].slice]
        eye_canthus_left = [x for x in left_eye if x[0] == max([x[0] for x in left_eye])][0]
        eye_canthus_right = [x for x in right_eye if x[0] == min([x[0] for x in right_eye])][0]
        get_temperature_n_mask(in_img, temp_csv, marker_thickness,
                               tuple(eye_canthus_left), tuple(eye_canthus_right))
    return in_img


def run_face_detection(dir_fp):
    for counter, file in enumerate(dir_fp.iterdir()):
        if 10 <= counter <= 160:
            print("Reading file - ", file)
            in_img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            temp_csv = pd.read_csv(
                str(TARGET_DIR / "TEMP_ARRAYS" / "{}.{}".format(file.stem, "csv")),
                header=None,
                sep=';'
            )
            # Method1
            print()
            out_img_v1 = ir_image_processing(in_img, temp_csv, MARKER_THICKNESS)
            # Method2
            print()
            out_imgv2 = ir_image_processing_v2(in_img, temp_csv, MARKER_THICKNESS)
            # Method3
            print()
            out_img_v3 = ir_image_processing_v3(in_img, temp_csv, MARKER_THICKNESS)

            if SAVE_RESULT:
                cv2.imwrite(
                    str(RESULTS_DIR / "{}.{}".format(file.stem, "jpg")),
                    np.hstack((in_img, out_img_v1, out_imgv2, out_img_v3))
                )
            else:
                # plot_image(in_img, out_img_v1, out_imgv2, out_img_v3)
                cv2.imshow(
                    "frame",
                    np.hstack((in_img, out_img_v1, out_imgv2, out_img_v3)))
                cv2.waitKey(1)
                cv2.destroyAllWindows()
        else:
            continue


def run():
    # create_data_dir()
    run_face_detection(TARGET_DIR / "IR_IMAGES")


if __name__ == '__main__':
    run()
