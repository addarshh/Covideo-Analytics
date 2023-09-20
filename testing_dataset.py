import random
import traceback
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import get_face_features, align_rgb_ir_image, face_add_eye_canthus, plot_image

DATASET_DIR = Path("datasets/RGB_Thermal_dataset")
MAX_SAMPLES = 113


def get_random_sample(sample=10):
    random_sample = random.sample(range(1, MAX_SAMPLES + 1), sample)
    sample_image_dict = {}

    for num in random_sample:
        sample_image_dict[num] = (
            DATASET_DIR / "{}_IR.jpg".format(num),
            DATASET_DIR / "{}_RGB.jpg".format(num),
        )

    return sample_image_dict


def read_sample_images(image_loc_dict):
    output_image_dict = {}
    try:
        for num, image_loc in image_loc_dict.items():
            ir_image, rgb_image = image_loc
            if ir_image.exists() and rgb_image.exists():
                print('Reading Image Num - ', num)
                output_image_dict[num] = (
                    cv2.imread(str(ir_image), cv2.IMREAD_COLOR),
                    cv2.imread(str(rgb_image), cv2.IMREAD_COLOR)
                )
            else:
                print('Aborting reading of Image - ', num)
    except Exception as ex:
        print('Failed due to - ', ex)
        traceback.format_exc()
    return output_image_dict


def color_image_processing(images_dict):
    proc_img_dict = {}
    for num, images_tuple in images_dict.items():
        ir_image, rgb_image = images_tuple
        new_width = int(rgb_image.shape[1] * (ir_image.shape[1] / rgb_image.shape[1]))
        new_height = int(rgb_image.shape[0] * (ir_image.shape[0] / rgb_image.shape[0]))
        print('Coverting image - {} from shape - {} to {}'.format(
            num,
            rgb_image.shape,
            (new_height, new_width, 3)))
        rgb_img_resize = cv2.resize(
            rgb_image,
            # (new_width, new_height),
            (int(rgb_image.shape[1] * 0.10),
             int(rgb_image.shape[0] * 0.10)),
            interpolation=cv2.INTER_LANCZOS4)
        ir_image = cv2.resize(
            ir_image,
            (int(rgb_image.shape[1] * 0.10),
             int(rgb_image.shape[0] * 0.10)),
            cv2.INTER_LANCZOS4)
        proc_img_dict[num] = (ir_image, rgb_image, rgb_img_resize)
    return proc_img_dict


def apply_transformation(images_dict):
    for num, images_tuple in images_dict.items():
        ir_image, rgb_image, rgb_image_resize = images_tuple
        print('Processing image number - ', num)
        ir_img_aligned, warp_matrix = align_rgb_ir_image(
            rgb_image_resize,
            ir_image,
            warp_mode=cv2.MOTION_EUCLIDEAN
        )

        face_location_list, face_landmark_list = get_face_features(
            rgb_image_resize
        )

        print("Found {} Faces in the image".format(len(face_location_list)))
        if len(face_location_list) == 0:
            continue
        for face_loc, face_landmark in zip(face_location_list, face_landmark_list):
            ir_img_aligned = cv2.rectangle(
                ir_img_aligned,
                (face_loc[3], face_loc[0]),  # left top
                (face_loc[1], face_loc[2]),  # right bottom
                color=(0, 255, 0), thickness=1)

            ir_img_aligned = cv2.polylines(
                ir_img_aligned, [np.array(face_landmark["left_eye"], dtype=np.int32)],
                color=(0, 255, 255), thickness=2, isClosed=False)
            ir_img_aligned = cv2.polylines(
                ir_img_aligned, [np.array(face_landmark["right_eye"], dtype=np.int32)],
                color=(0, 255, 255), thickness=2, isClosed=False)

            rgb_image_resize = cv2.rectangle(
                rgb_image_resize,
                (face_loc[3], face_loc[0]),  # left top
                (face_loc[1], face_loc[2]),  # right bottom
                color=(0, 255, 0), thickness=1)

            rgb_image_resize = cv2.polylines(
                rgb_image_resize, [np.array(face_landmark["left_eye"], dtype=np.int32)],
                color=(0, 255, 255), thickness=2, isClosed=False)
            rgb_image_resize = cv2.polylines(
                rgb_image_resize, [np.array(face_landmark["right_eye"], dtype=np.int32)],
                color=(0, 255, 255), thickness=2, isClosed=False)

            plt.imshow(cv2.cvtColor(
                np.hstack([rgb_image_resize, ir_img_aligned, ir_image]), cv2.COLOR_BGR2RGB))
            plt.show()


def test_eye_canthus_detection(images_dict):
    for num, images_tuple in images_dict.items():
        ir_image, rgb_image, rgb_image_resize = images_tuple
        print('Processing image number - ', num)
        rgb_image_resize = face_add_eye_canthus(rgb_image_resize, region_thickness=10)
        plot_image(rgb_image_resize)
        plt.show()


if __name__ == '__main__':
    # random.seed(123)
    images_sample = get_random_sample(10)
    print('Number of samples extracted - ', len(images_sample))
    images_sample = read_sample_images(images_sample)
    print("Number of images read - ", len(images_sample))

    img_proc_dict = color_image_processing(images_sample)

    apply_transformation(img_proc_dict)
    # test_eye_canthus_detection(img_proc_dict)
