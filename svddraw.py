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


def do_bootstrap(d, md, ui, gt_landmarks):
    test_initial_guess = np.concatenate([[0, 0, 0, 0.001, 0.001, -2, 1], np.zeros(46)])
    bound_bfgs = [(-2, 2)] * 3 + [(None, None)] * 2 + [(None, -1)] + [(0, None)] + [(0, 1)] * 46
    test_bs = get_blend_shape(d, np.average(ui, axis=0))
    guess_frame0, e_frame0, info = op.fmin_l_bfgs_b(cost.f_sq, test_initial_guess, bounds=bound_bfgs, approx_grad=True,
                                                    args=(cost.f_p_f_2, md, gt_landmarks[0], test_bs, 0.01))
    print("e_frame0={} f={}".format(e_frame0, guess_frame0[6]))
    guess_frames = [guess_frame0]
    for i in range(1, gt_landmarks.shape[0]):
        # ig = np.average(guess_frames, axis=0)
        gf, ef, ei = op.fmin_l_bfgs_b(cost.f_sq, test_initial_guess, bounds=bound_bfgs,
                                      approx_grad=True,
                                      args=(cost.f_p_f_2, md, gt_landmarks[i], test_bs, 0.01))
        print("e_frame{}={} f={} r={} t={}".format(i, ef, gf[6], gf[0:3], gf[3:6]))
        guess_frames.append(gf)

    return guess_frames[0]


def build_landmark_img(dx, w):
    hw = w // 2
    img = np.zeros((w, w, 3), np.uint8)
    for i in range(dx.shape[0]):
        cv2.circle(img, (int(-dx[i, 0] * hw + hw), int(-dx[i, 1] * hw + hw)), 2, (255, 0, 255))
    return img


def test():
    focus_length = 0.8
    min_keyframes = 3
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

    def fb(v_id):
        return get_blend_shape(d, v_id)

    cam_landmarks = np.fromfile("C:\\dev\\3dface\\keyframes\\landmarks", dtype=np.float32, count=min_keyframes * 68 * 2) \
        .reshape(
        (min_keyframes, 68, 2))

    # initial_guess, bounds = cost.build_initial_guess_and_bound(min_keyframes, ui[0], 47)
    # print(initial_guess)

    # gt_landmarks = test_gt_id8_exp7
    # r = op.least_squares(cost.f_id_f, initial_guess, bounds=bounds, verbose=2, x_scale='jac',
    #                      args=(md, gt_landmarks, min_keyframes, ui[0].shape[0], fb, 0.001))
    # print(r.x)
    # print(r.fun)

    # test_bs = get_blend_shape(d, ui[8])
    # gt_landmarks = dx
    # print(gt_landmarks.shape)
    # initial_e_guess = np.zeros(46)
    # initial_e_guess[6] = 0

    # test_initial_guess = np.concatenate([[1, 1, 1, 0.001, 0.001, -3], initial_e_guess])
    # test_bound_lower = np.concatenate([[-2, -2, -2, -np.inf, -np.inf, -np.inf], np.zeros(46)])
    # test_bound_upper = np.concatenate([[2, 2, 2, np.inf, np.inf, 0], np.ones(46)])
    #
    # r = op.least_squares(cost.f_p, test_initial_guess, verbose=2, method="trf",
    #                      bounds=(test_bound_lower, test_bound_upper),
    #                      args=(md, gt_landmarks, focus_length, test_bs, 0.001))
    #
    # print(r.x)
    # print(r.grad)
    # print(r.jac)



    guess_frame0 = do_bootstrap(d, md, ui, cam_landmarks * (-1))
    guess_r3 = guess_frame0[0:3]
    guess_t3 = guess_frame0[3:6]
    guess_f = guess_frame0[6]
    guess_e = guess_frame0[7:]

    guess_e = np.concatenate([np.array(1 - np.sum(guess_e))[None], guess_e])
    rv = vt.rt(vec_to_xyzw(guess_e @ get_blend_shape(c, np.average(ui, axis=0)) + m), *guess_t3, *guess_r3).transpose()
    draw.write_parameters(0, rv)
    draw.write_parameters(3, guess_f)

    dx = cam_landmarks[0] * -1
    landmark_image = build_landmark_img(dx, 600)
    draw.write_parameters(4, landmark_image)
    draw.write_parameters(6, 1)

    draw.start_render_window_thread(1200)


    # cap = cv2.VideoCapture(1)
    # ret, frame = cap.read()
    # if frame is None:
    #     raise Exception('摄像头坏了。')
    # h, w, c = frame.shape
    # draw.write_parameters(6, w / h)
    # print(w, h)
    # long_edge = w
    # if h > w:
    #     long_edge = h
    # while True:
    #     ret, frame = cap.read()
    #     if frame is None:
    #         raise Exception('摄像头坏了。')
    #     img = frame.copy()
    #     d = landmark_detector.detect_landmark_68(img)
    #     if d is not None:
    #         for i in range(68):
    #             x, y = d[i]
    #             cv2.circle(img, (x, y), 3, (0, 255, 255))
    #             cv2.putText(img, "{}".format(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    #         if len(keyframes) < min_keyframes:
    #             keyframes.append(landmark_detection_to_screen_xy(d, w, h))
    #             idx = len(keyframes)
    #             cv2.imwrite("C:\\dev\\3dface\\keyframes\\{}.jpg".format(idx), img)
    #             print(idx)
    #             if idx == min_keyframes:
    #                 a = np.array(keyframes)
    #                 a.tofile("C:\\dev\\3dface\\keyframes\\landmarks")
    #                 print(a.dtype, a.shape)
    #
    #     # for i in landmark:
    #     #     sc = screen_xyzw_to_pixel(scx[i], long_edge // 2, w // 2, h // 2)
    #     #     cv2.circle(img, sc, 2, (255, 0, 255))
    #     # for i in range(dx.shape[0]):
    #     #     sc = screen_xyzw_to_pixel(dx[i], long_edge // 2, w // 2, h // 2)
    #     #     cv2.circle(img, sc, 2, (255, 0, 255))
    #     #     cv2.putText(img, "{}".format(i), sc, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    #     draw.write_parameters(4, img)


if __name__ == '__main__':
    test()
