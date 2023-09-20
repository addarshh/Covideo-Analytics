import imutils
from utils import face_add_eye_canthus, plot_image, get_point_temp
import random
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
RESULT_DIR = Path("datasets/E95_dataset_ir_optic/RESULTS")
if not RESULT_DIR.exists():
    RESULT_DIR.mkdir()
else:
    for f in RESULT_DIR.iterdir():
        os.remove(str(f))

RESIZE_FACTOR = 0.8
DO_VISUAL = False
START, STOP, NUM_PTS = 0.6, 0.8, 5

for file in os.listdir("datasets/E95_dataset_ir_optic/IR_IMAGES"):
    # ir_random_file = random.choice(os.listdir("datasets/E95_dataset_ir_optic/IR_IMAGES"))
    ir_random_file = file
    filename = ir_random_file.split('.')[0]
    print("Reading all the image related to - ", ir_random_file)
    try:
        ir_img = cv2.imread("datasets/E95_dataset_ir_optic/IR_IMAGES/{}".format(ir_random_file),
                            cv2.IMREAD_GRAYSCALE)
        temp_arr = pd.read_csv(
            "datasets/E95_dataset_ir_optic/TEMP_ARRAYS/{}.csv".format(
                ir_random_file.split(".")[0]),
            header=None, sep=";")

        rgb_img = cv2.imread("datasets/E95_dataset_ir_optic/RGB_IMAGES/{}_optical_axis.jpg".format(
            ir_random_file.split(".")[0]))
    except FileNotFoundError:
        print("WARNING - All the files not found for - ", filename)
        continue

    start = time.time()
    rgb_img_resize = cv2.resize(
        rgb_img,
        (int(rgb_img.shape[1] * RESIZE_FACTOR), int(rgb_img.shape[0] * RESIZE_FACTOR)),
        cv2.INTER_AREA)
    rgb_img_resize_gry = cv2.cvtColor(rgb_img_resize, cv2.COLOR_BGR2GRAY)

    # print("All the image sizes - ", rgb_img.shape, rgb_img_resize.shape, ir_img.shape)

    # print('Creating template using IR image...')
    template = cv2.Canny(ir_img, 25, 100)
    (tH, tW) = template.shape[:2]
    # print('using template of shape - ', (tW, tH))
    if DO_VISUAL:
        plot_image(template)

    found = None
    gray = rgb_img_resize_gry.copy()
    for scale in np.linspace(START, STOP, NUM_PTS)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        # print("Resized image shape - ", resized.shape)
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        # print("Ratio of resizing - ", r)
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # resized = cv2.GaussianBlur(resized, (5, 5), 0)
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        # print("Matching template result in - {} at loc - {}".format(maxVal, maxLoc))
        ir_img_dup = ir_img.copy()
        # check to see if the iteration should be visualized
        x1, x2 = maxLoc[1], maxLoc[1] + tH
        y1, y2 = maxLoc[0], maxLoc[0] + tW
        crop_img = resized[x1:x2, y1:y2].copy()

        if DO_VISUAL:
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            plot_image(crop_img, ir_img_dup)
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                          (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)
        if found is None or maxVal > found[0]:
            # if len(face_list) > 0:
            found = (maxVal, ir_img_dup, crop_img)
            # else:
            #     pass

    if found is None:
        print("Not able to detect face while templeate matching...")
    else:
        (_, ir_img_dup, crop_img) = found
        crop_img, face_list, canthus_list = face_add_eye_canthus(crop_img, region_thickness=3, get_feature=True)
        # print(crop_img, face_list, canthus_list)
        for face in face_list:
            x1, x2 = face[0], face[2]
            y1, y2 = face[3], face[1]
            ir_img_face = ir_img_dup.copy()[x1:x2, y1:y2]
            ir_img_face = cv2.GaussianBlur(ir_img_face, (15, 15), 0)
            (min_Val, max_Val, min_Loc, max_Loc) = cv2.minMaxLoc(ir_img_face)
            pt_x, pt_y = face[3] + max_Loc[0], face[0] + max_Loc[1]

            ir_img_dup = cv2.circle(ir_img_dup, (pt_x, pt_y), 3, (0, 255, 255), 3)
            crop_img = get_point_temp(crop_img, temp_arr.values, (pt_x, pt_y))
            ir_img_dup = cv2.rectangle(
                ir_img_dup,
                (face[3], face[0]),  # left top
                (face[1], face[2]),  # right bottom
                color=(0, 255, 0),
                thickness=3)
        # for left, right in canthus_list:
        #     ir_img_dup = cv2.circle(ir_img_dup, left, 3, (0, 255, 255), 1)
        #     ir_img_dup = cv2.circle(ir_img_dup, right, 3, (0, 255, 255), 1)

        cv2.imwrite(
            str(RESULT_DIR / "{}.jpg".format(filename)),
            np.hstack((crop_img, ir_img_dup)))
    print("File processed in - {} secs".format(np.round(time.time() - start, 1)))
