import fwmodel
import cv2
import tensor as T
import numpy as np
import draw
import fwmesh
import landmark_detector
import vertex_screen as vt
import cost
import scipy.optimize as op
import time


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


def landmark_detection_to_screen_xy(d, w, h):
    le = w
    if h > w:
        le = h
    d = d - np.array([w // 2, h // 2])
    a = np.divide(d, le / 2, dtype=np.float32)
    return a


def truncate_core_tensor(core_tensor, mean_vertex_axis, landmarks):
    c = landmarks * 3
    index = np.stack((c, c + 1, c + 2)).transpose().flatten()
    return core_tensor[:, :, index], mean_vertex_axis[index]


def build_test_gt_landmark(vv, t3r3, focus_length):
    a = np.array([vt.proj(vt.rt(vec_to_xyzw(vv), *(t3r3[i])), focus_length).transpose()
                  for i in range(t3r3.shape[0])])
    return a[:, :, 0:2] / a[:, :, 3][:, :, None]


def solve_rtes_frame(test_bs, rte, f, stopcond, md, gt_landmarks):
    bound_bfgs = [(-2, 2)] * 3 + [(None, None)] * 2 + [(None, -1)] + [(0, 1)] * 46
    guess_frame0, e_frame0, info = op.fmin_l_bfgs_b(cost.f_sq, rte[0], bounds=bound_bfgs, approx_grad=True,
                                                    factr=stopcond,
                                                    args=(cost.f_p_2, md, gt_landmarks[0], f, test_bs, 0.01))
    print("e_frame0={} r={} t={}".format(e_frame0, guess_frame0[0:3], guess_frame0[3:6]))
    guess_frames = [guess_frame0]
    e_frames = [e_frame0]
    for i in range(1, gt_landmarks.shape[0]):
        gf, ef, ei = op.fmin_l_bfgs_b(cost.f_sq, rte[i], bounds=bound_bfgs, factr=stopcond,
                                      approx_grad=True,
                                      args=(cost.f_p_2, md, gt_landmarks[i], f, test_bs, 0.01))
        print("e_frame{}={} r={} t={}".format(i, ef, gf[0:3], gf[3:6]))
        guess_frames.append(gf)
        e_frames.append(ef)
    print(np.sum(e_frames))
    return guess_frames


def do_one_iteration(u, rte, f, d, md, gt_landmarks):
    def fb(v_id):
        return get_blend_shape(d, v_id)

    r = op.least_squares(cost.f_id, u, verbose=2, x_scale='jac', method="lm",
                         args=(md, gt_landmarks, f, np.array(rte), fb, 0.1))

    print(r.x)
    test_bs = get_blend_shape(d, r.x)
    guess_frames = solve_rtes_frame(test_bs, rte, f, 1e9, md, gt_landmarks)
    return r.x, guess_frames


def do_bootstrap(d, md, ui, gt_landmarks):
    f_max = 5  # fov 23
    f_min = 0.25  # fov 150
    f = 2
    initial_u_guess = np.average(ui, axis=0)
    initial_rte_guess = [np.concatenate([[0, 0, 0, 0.001, 0.001, -2], np.zeros(46)])] * gt_landmarks.shape[0]
    test_bs = get_blend_shape(d, initial_u_guess)
    guess_frames = solve_rtes_frame(test_bs, initial_rte_guess, f, 1e9, md, gt_landmarks)

    u = initial_u_guess
    rte = guess_frames
    for i in range(3):
        u, rte = do_one_iteration(u, rte, f, d, md, gt_landmarks)

    return u, rte


def build_landmark_img(dx, w):
    hw = w // 2
    img = np.zeros((w, w, 3), np.uint8)
    for i in range(dx.shape[0]):
        cv2.circle(img, (int(-dx[i, 0] * hw + hw), int(-dx[i, 1] * hw + hw)), 2, (255, 0, 255))
    return img


def render_model(c, u, m, guess_frame):
    guess_frame0 = guess_frame
    guess_r3 = guess_frame0[0:3]
    guess_t3 = guess_frame0[3:6]
    guess_e = guess_frame0[6:]

    guess_e = np.concatenate([np.array(1 - np.sum(guess_e))[None], guess_e])
    rv = vt.rt(vec_to_xyzw(guess_e @ get_blend_shape(c, u) + m), *guess_t3,
               *guess_r3).transpose()
    draw.write_parameters(0, rv)


def test():
    focus_length = 0.8
    min_keyframes = 40
    keyframes = []
    vertex, triangle, landmark = fwmesh.read_mesh_def()
    draw.write_parameters(0, vertex)
    draw.write_parameters(1, triangle)
    draw.write_parameters(2, landmark)
    draw.write_parameters(3, focus_length)
    draw.write_parameters(5, 0.7)
    draw.write_parameters(6, 1.33)

    se, ue, si, ui, c, m = fwmodel.load_compact_svd('C:\\dev\\3dface\\svd2', 40, 47)
    c = T.mode_dot(c, ue, 1)  # we don't need SVD on exp axis

    d, md = truncate_core_tensor(c, m, landmark)
    vv = get_blend_shape(d, ui[8])[7] + md
    test_gt_id8_exp7 = build_test_gt_landmark(vv, np.array([[1, 1, -2, 0.1, 0.1, 0.1],
                                                            [-0.5, 1, -2, -0.1, 0.1, 0.1],
                                                            [1, 0, -4, 0.1, 0.2, -0.1],
                                                            [1, 0, -3, 0.1, 0.1, 0.1],
                                                            [1, 0, -3, 0.1, 0.1, 0.1]
                                                            ]), focus_length)
    dx = test_gt_id8_exp7[0]
    print(dx[0])

    cam_landmarks = np.fromfile("C:\\dev\\3dface\\keyframes\\landmarks", dtype=np.float32, count=min_keyframes * 68 * 2) \
        .reshape(
        (min_keyframes, 68, 2))

    draw.start_render_window_thread(1200)
    u, gs = do_bootstrap(d, md, ui, cam_landmarks * (-1))
    test_bs = get_blend_shape(d, u)
    # for i in range(min_keyframes):
    #     render_model(c, u, m, gs[i])
    #     draw.write_parameters(3, 2)
    #
    #     dx = cam_landmarks[i] * -1
    #     landmark_image = build_landmark_img(dx, 600)
    #     draw.write_parameters(4, landmark_image)
    #     draw.write_parameters(6, 1)
    #     draw.refresh_display()
    #     time.sleep(0.5)


    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if frame is None:
        raise Exception('摄像头坏了。')
    h, w, channel = frame.shape
    draw.write_parameters(6, w / h)
    print(w, h)
    long_edge = w
    if h > w:
        long_edge = h
    current_guess = np.array([np.concatenate([[0, 0, -2, 0.001, 0.001, 0], np.zeros(46)])])
    f = 2
    draw.write_parameters(3, f)
    while True:
        ret, frame = cap.read()
        if frame is None:
            raise Exception('摄像头坏了。')
        img = frame.copy()
        d = landmark_detector.detect_landmark_68(img)
        if d is not None:
            for i in range(68):
                x, y = d[i]
                cv2.circle(img, (x, y), 3, (0, 255, 255))
                cv2.putText(img, "{}".format(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            # if len(keyframes) < min_keyframes:
            #     keyframes.append(landmark_detection_to_screen_xy(d, w, h))
            #     idx = len(keyframes)
            #     cv2.imwrite("C:\\dev\\3dface\\keyframes\\{}.jpg".format(idx), img)
            #     print(idx)
            #     if idx == min_keyframes:
            #         a = np.array(keyframes)
            #         a.tofile("C:\\dev\\3dface\\keyframes\\landmarks")
            #         print(a.dtype, a.shape)
            gt_frame = landmark_detection_to_screen_xy(d, w, h) * -1

            current_guess = solve_rtes_frame(test_bs, current_guess, f, 1e9, md, gt_frame[None, :, :])
            render_model(c, u, m, current_guess[0])

        # for i in landmark:
        #     sc = screen_xyzw_to_pixel(scx[i], long_edge // 2, w // 2, h // 2)
        #     cv2.circle(img, sc, 2, (255, 0, 255))
        # for i in range(dx.shape[0]):
        #     sc = screen_xyzw_to_pixel(dx[i], long_edge // 2, w // 2, h // 2)
        #     cv2.circle(img, sc, 2, (255, 0, 255))
        #     cv2.putText(img, "{}".format(i), sc, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        draw.write_parameters(4, img)
        draw.refresh_display()


if __name__ == '__main__':
    test()
