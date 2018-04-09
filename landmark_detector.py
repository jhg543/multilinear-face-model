import os
import dlib
import glob
import cv2
import numpy as np


class DLibLandmarkTracker:
    def __init__(self, predictor_path, detect_every_frame=False):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.last_bbox = None
        self.detect_every_frame = detect_every_frame

    def detect(self, img):
        r = self.__detect(img)
        if r is None:
            self.last_bbox = None
        else:
            self.last_bbox = (np.min(r[:, 0]), np.max(r[:, 0]), np.min(r[:, 1]), np.max(r[:, 1]))
        return r

    def __detect(self, img):
        h, w, nc = img.shape
        if self.last_bbox is None or self.detect_every_frame:
            dets = self.detector(img, 1)
            if len(dets) != 1:
                return None
            for k, d in enumerate(dets):
                top, bottom, left, right = (d.top(), d.bottom(), d.left(), d.right())
        else:
            l, r, t, b = self.last_bbox
            dy = (b - t) // 20
            dx = (r - l) // 20
            top, bottom, left, right = (max((0, t - dy)), min((h - 1, b + dy)), max((0, l - dx)), min((w - 1, r + dx)))

        dlib_bbox = dlib.rectangle(left, top, right, bottom)
        h, w, c = img.shape
        if top < 3 or bottom > h - 3 or left < 3 or right > w - 3:
            return None
        shape = self.predictor(img, dlib_bbox)
        if shape.num_parts != 68:
            return None
        # Draw the face landmarks on the screen.
        r = np.empty((68, 2), dtype=np.int32)
        for i in range(68):
            a = shape.part(i)
            r[i] = (a.x, a.y)
            if a.y < 3 or a.y > h - 3 or a.x < 3 or a.x > w - 3:
                return None
        current_bbox = (np.min(r[:, 0]), np.max(r[:, 0]), np.min(r[:, 1]), np.max(r[:, 1]))
        if self.last_bbox is not None:
            ct, cb, cl, cr = current_bbox
            rt = (cb - ct) / (0.01 + cr - cl)  # avoid div by 0
            diff = max([abs(x) for x in [ct - top, cb - bottom, cl - left, cr - right]]) // (bottom - top)
            if rt > 1.5 or rt < 0.7 or diff > 0.1:
                return None
        return r


if __name__ == '__main__':
    faces_folder_path = "C:\\dev\\3dface\\tmpimg"
    win = dlib.image_window()
    dd = DLibLandmarkTracker("C:\\dev\\3dface\\landmark\\shape_predictor_68_face_landmarks.dat", True)
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = cv2.imread(f)
        win.clear_overlay()
        d = dd.detect(img)
        for i in range(68):
            x = d[i, 0]
            y = d[i, 1]
            cv2.circle(img, (x, y), 5, (255, 255, 255))
            cv2.putText(img, "{}".format(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        win.set_image(img)
        dlib.hit_enter_to_continue()
