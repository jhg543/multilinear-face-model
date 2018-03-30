import os
import dlib
import glob
import cv2
import numpy as np

predictor_path = "C:\\dev\\3dface\\landmark\\shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def detect_landmark_68(img):
    dets = detector(img, 1)
    if len(dets) != 1:
        return None
    for k, d in enumerate(dets):
        h, w, c = img.shape
        if d.top() < 3 or d.bottom() > h - 3 or d.left() < 3 or d.right() > w - 3:
            return None
        shape = predictor(img, d)
        if shape.num_parts != 68:
            return None
        # Draw the face landmarks on the screen.
        r = np.empty((68, 2), dtype=np.int32)
        for i in range(68):
            a = shape.part(i)
            r[i] = (a.x, a.y)
        return r


if __name__ == '__main__':
    faces_folder_path = "C:\\dev\\3dface\\tmpimg"
    win = dlib.image_window()
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = cv2.imread(f)
        win.clear_overlay()
        d = detect_landmark_68(img)
        for i in range(68):
            x = d[i, 0]
            y = d[i, 1]
            cv2.circle(img, (x, y), 5, (255, 255, 255))
            cv2.putText(img, "{}".format(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        win.set_image(img)
        dlib.hit_enter_to_continue()
