import fwmodel
import cv2
import tensor as T
import numpy as np
import draw
import fwmesh
import landmark_detector
import vertex_screen


def test():
    focus_length = 0.8
    min_keyframes = 20
    keyframes = []
    vertex, triangle, landmark = fwmesh.read_mesh_def()
    draw.write_parameters(0, vertex)
    draw.write_parameters(1, triangle)
    draw.write_parameters(2, landmark)
    draw.write_parameters(3, focus_length)
    draw.write_parameters(5, 0.7)
    draw.write_parameters(6, 1.33)
    draw.start_render_loop_thread()
    se, ue, si, ui, c, m = fwmodel.load_compact_svd('C:\\dev\\3dface\\svd2', 40, 47)
    c = T.mode_dot(c, ue, 1)  # we don't need SVD on exp axis

    def get_blend_shape(core_id, v_id):
        return T.mode_dot(core_id, np.expand_dims(v_id, axis=0), 0)[0]

    def vec_to_xyzw(v):
        x = v.shape[0] // 3
        r = np.ones((x, 4), dtype=v.dtype)
        r[:, :-1] = v.reshape((-1, 3))
        return r.transpose()

    def screen_xyzw_to_pixel(v, l, cw, ch):
        x, y = v[0:2] * l / v[3]
        return (-int(x) + cw, -int(y) + ch)

    rv = vertex_screen.rt(vec_to_xyzw(get_blend_shape(c, ui[8])[7] + m), 0, 0, -2, 0.1, 0.1,
                          0.1).transpose()
    draw.write_parameters(0, rv)
    print(rv[landmark[0]])
    scx = vertex_screen.proj(rv.transpose(), focus_length).transpose()
    print(scx[landmark[0]])
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if frame is None:
        raise Exception('摄像头坏了。')
    h, w, c = frame.shape
    draw.write_parameters(6, w / h)
    print(w, h)
    long_edge = w
    if (h > w):
        long_edge = h
    while True:
        ret, frame = cap.read()
        if frame is None:
            raise Exception('摄像头坏了。')
        img = frame.copy()
        d = landmark_detector.detect_landmark_68(img)
        if d is not None:
            if len(keyframes) < min_keyframes:
                keyframes.append(d)
            for i in range(68):
                x, y = d[i]
                cv2.circle(img, (x, y), 3, (0, 255, 255))
                # cv2.putText(img, "{}".format(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        for i in landmark:
            sc = screen_xyzw_to_pixel(scx[i], long_edge // 2, w // 2, h // 2)
            cv2.circle(img, sc, 2, (255, 0, 255))
        draw.write_parameters(4, img)


if __name__ == '__main__':
    test()
