import os
from zipfile import ZipFile
from pathlib import Path
from shutil import copyfile, rmtree
import tempfile
import traceback
import shutil
import cv2
import numpy as np
from utils import face_add_eye_canthus, plot_image

# from flir_e95_processing import ir_image_processing

ir_image_fp = "datasets\\E95_dataset\\IR_IMAGES\\2020-03-26T001823102_src1.jpg"
ir_img = cv2.imread(ir_image_fp, cv2.IMREAD_GRAYSCALE)
# ir_img_process = ir_image_processing(ir_img)

in_img = ir_img.copy()
kernel = np.array(
    [[0, 0, -2, 0, 0],
     [0, -2, -2, -2, 0],
     [-2, -2, 25, -2, -2],
     [0, -2, -2, -2, 0],
     [0, 0, -2, 0, 0]])
in_img = cv2.GaussianBlur(in_img, (15, 15), 0)
# in_img = cv2.filter2D(in_img, -1, kernel)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(in_img)
print(minVal, maxVal, minLoc, maxLoc)
(x, y) = maxLoc
out_image = cv2.circle(in_img, maxLoc, 3, (0, 255, 255), 3)
# cv2.rectangle(image, (maxLoc), (390, 190), (0, 255, 0), 2)

# out_image = face_add_eye_canthus(out_img, region_thickness=3)
plot_image(ir_img, out_image)
from utils import get_face_features, align_rgb_ir_image

ir_img = cv2.imread("datasets/E95_dataset_ir_optic/IR_IMAGES/2020-03-26T235958796_src1.jpg",
                    cv2.IMREAD_GRAYSCALE)

rgb_img = cv2.imread("datasets/E95_dataset_ir_optic/RGB_IMAGES/2020-03-26T235958796_src1_optical_axis.jpg")

# For RGB image
mask = np.zeros(rgb_img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

face_list, feace_feat_list = get_face_features(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY))
face_loc = face_list[0]
# (top, right, bottom, left) order
PIXEL_OFFSET_UP, PIXEL_OFFSET_BELOW = 200, 30
x1, y1 = max(face_loc[3] - PIXEL_OFFSET_UP, 0), \
         max(face_loc[0] - PIXEL_OFFSET_UP, 0)
x2, y2 = min(face_loc[1] + PIXEL_OFFSET_BELOW, rgb_img.shape[1]), \
         min(face_loc[2] + PIXEL_OFFSET_BELOW, rgb_img.shape[0])
print(x1, y1, x2, y2)
cv2.grabCut(rgb_img, mask, (x1, y1, x2, y2), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
rgb_img_2 = rgb_img * mask2[:, :, np.newaxis]
plot_image(rgb_img_2, rgb_img)

# For IR images
_, threshold = cv2.threshold(ir_img, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plot_image(threshold)

plot_image(mask2)

rgb_img_2_gray = cv2.cvtColor(rgb_img_2, cv2.COLOR_BGR2GRAY)
plot_image(rgb_img_2_gray)
plot_image(cv2.circle(mask, max_loc, 5, 0, thickness=3))

_, _, _, max_loc = cv2.minMaxLoc(mask)

x1, x2 = max_loc[1] - ir_img.shape[1] // 2, max_loc[1] + (ir_img.shape[1] - ir_img.shape[1] // 2)
y1, y2 = max_loc[0] - ir_img.shape[0] // 2, max_loc[0] + (ir_img.shape[1] - ir_img.shape[1] // 2)

rgb_img_2_al = align_rgb_ir_image(ir_img, rgb_img_2, cv2.MOTION_AFFINE)
plot_image(rgb_img_2_al)

#################################################################################################
