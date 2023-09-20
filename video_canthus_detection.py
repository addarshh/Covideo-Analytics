import cv2
import numpy as np
import time
from utils import face_add_eye_canthus, plot_image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.)

    # frame_resize = cv2.resize(
    #     frame, (336, 256),
    #     interpolation=cv2.INTER_AREA)
    frame_canthus, _, _ = face_add_eye_canthus(frame, region_thickness=5, get_feature=True)
    cv2.imshow("frame", frame_canthus)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# cap = cv2.VideoCapture("datasets/FLIR1215.mp4")
#
# kernel = np.array([[0, -2, 0],
#                    [-2, 9, -2],
#                    [0, -2, 0]])
#
# # ret, frame = cap.read()
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame_gray_smooth = cv2.GaussianBlur(
#             frame_gray, (3, 3), 0.5)
#         # frame_gray = cv2.filter2D(frame_gray, -1, kernel)
#         print(frame.shape, frame_gray.shape)
#         frame_edge = cv2.Canny(
#             frame_gray, 150, 220)
#         # plot_image()
#         # plot_image(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
#         # plot_image(face_add_eye_canthus(frame_gray))
#         cv2.imshow("frame", frame)
#         cv2.imshow("frame gray", frame_gray)
#         cv2.imshow("frame edge", frame_edge)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
#     time.sleep(0.5)
#
# cap.release()
# cv2.destroyAllWindows()
