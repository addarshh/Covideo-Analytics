"""
Function to include following -
1. Face Feature Extraction
2. Mapping Color (RGB) image to IR (thermal image)
"""

import cv2
import face_recognition
import numpy as np
from matplotlib import pyplot as plt


def get_face_features(img_array, fd_model="hog", feature_model="large", verbose=False):
    """
    Function to get the face detection and facial features
    :param img_array: Input image array
    :param fd_model: Face detection model to be used. either "hog" or "cnn"
    :param feature_model: Feature detection model to be used. either "small" or "large"
    :param verbose: Flag for showing the print statements
    :return: (face coordinates list, face landmarks list) else (None, None)
    """
    if verbose:
        print('Processing image shape {} to find faces....'.format(img_array.shape))
    face_location = face_recognition.face_locations(img_array, model=fd_model)
    if verbose:
        print("Found {} face(s) in the image!!!.".format(len(face_location)))
    if len(face_location) == 0:
        return None, None
    face_landmarks_list = face_recognition.face_landmarks(img_array, model=feature_model)
    return face_location, face_landmarks_list


def align_rgb_ir_image(rgb_img, ir_image, warp_mode=cv2.MOTION_AFFINE, num_iteration=5000, termination_eps=1e-8):
    """
    Function will try to align the IR image accoding to the color image
    :param rgb_img: Input array of RGB image
    :param ir_image: Input array of IR(thermal) image
    :param warp_mode:
    :param termination_eps:
    :param num_iteration:
    :return: (Aligned IR image, warp matrix)
    """
    im1, im2 = rgb_img, ir_image
    im1_gray = im1 if len(im1.shape) == 2 else cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = im2 if len(im2.shape) == 2 else cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Find size of image1
    sz = im1.shape
    # Define the motion model
    warp_mode = warp_mode  # we can select cv2.MOTION_HOMOGRAPHY or cv2.MOTION_AFFINE
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    # elif warp_mode == cv2.MOTION_TRANSLATION:
    #     warp_matrix = np.eye(2, 2, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iteration, termination_eps)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im2_aligned, warp_matrix


def plot_image(*images):
    """
    Function to plot many images simultaneously
    :param images: args of the images
    :return: None, show the plots
    """
    img_shapes_set = set([x.shape for x in images])
    if len(images) == 1:
        # print(images[0].shape, images[0].dtype)
        if len(images[0].shape) == 3:
            # For color images
            plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
        else:
            # For grayscale images
            plt.imshow(images[0], cmap="gray")
        plt.show()
    else:
        if len(set([x.shape for x in images])) == 1:
            # print('Concatenating the images..')
            unq_shape = list(img_shapes_set)[0]
            out = np.uint8(images[0]) if len(images) == 1 else np.hstack(images)
            # print(out.dtype, out.shape, unq_shape)
            if len(unq_shape) == 3:
                # For color images
                plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            else:
                # For grayscale images
                plt.imshow(out, cmap="gray")
            plt.show()
        else:
            print("Images Size different...Plotting images Separately...")
            for img in images:
                img_type = "gray" if len(img.shape) == 2 else None
                if img_type == "gray":
                    plt.imshow(img, cmap="gray")
                else:
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()


def face_detect_add_features(image, *feats_to_display):
    image = np.copy(image)
    faces_location_list, faces_landmark_list = get_face_features(image, "cnn", "small")
    if faces_location_list is None:
        print('WARN - No faces found in the image')
        return image
    for face, features in zip(faces_location_list, faces_landmark_list):
        image = cv2.rectangle(
            image,
            (face[3], face[0]),  # left top
            (face[1], face[2]),  # right bottom
            color=(0, 255, 0),
            thickness=1)

        for feat in feats_to_display:
            if feat in features:
                image = cv2.polylines(
                    image,
                    [np.array(features[feat], dtype=np.int32)],
                    color=(0, 255, 255),
                    thickness=2,
                    isClosed=False)
            else:
                print("WARN - No detection found for feature - ", feat)
    return image


def face_add_eye_canthus(image, region_thickness=1, get_feature=False):
    image = np.copy(image)
    faces_location_list, faces_landmark_list = get_face_features(image)
    eye_canthus_list, face_list = [], []
    if faces_location_list is None:
        print('WARN - No faces found in the image')
        return image, face_list, eye_canthus_list
    for face, features in zip(faces_location_list, faces_landmark_list):
        image = cv2.rectangle(
            image,
            (face[3], face[0]),  # left top
            (face[1], face[2]),  # right bottom
            color=(0, 255, 0),
            thickness=region_thickness)
        left_eye, right_eye = features['left_eye'], features['right_eye']
        eye_canthus_left = [x for x in left_eye if x[0] == max([x[0] for x in left_eye])]
        eye_canthus_right = [x for x in right_eye if x[0] == min([x[0] for x in right_eye])]

        image = cv2.circle(image, eye_canthus_left[0], 3, (0, 255, 255), region_thickness)
        image = cv2.circle(image, eye_canthus_right[0], 3, (0, 255, 255), region_thickness)
        eye_canthus_list.append((eye_canthus_left[0], eye_canthus_right[0]))
    if get_feature:
        return image, faces_location_list, eye_canthus_list
    else:
        return image


def get_mask_box(point, num_pix_above, num_pix_below=None):
    num_pix_below = num_pix_below if num_pix_below is not None else num_pix_above
    top_left = (point[0] - num_pix_above, point[1] - num_pix_above)
    bottom_right = (point[0] + num_pix_below, point[1] + num_pix_below)
    return top_left, bottom_right


def get_temp_in_celcius(temp):
    return np.round((temp * (9 / 5)) + 32, 1)


def get_point_temp(rgb_img, temp_arr, point):
    top_left, bottom_right = get_mask_box(point, num_pix_above=5)
    temp_selected = temp_arr[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    min_temp, max_temp, avg_temp = \
        get_temp_in_celcius(np.min(temp_selected)), \
        get_temp_in_celcius(np.max(temp_selected)), \
        get_temp_in_celcius(np.mean(temp_selected))
    text_str = "{}F | {}F".format(avg_temp, max_temp)
    # temp_arr = cv2.circle(, point, 3, (255, 255, 255), 3)
    img = cv2.putText(
        rgb_img, text_str,
        (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        255, 2, cv2.LINE_AA)
    return img
