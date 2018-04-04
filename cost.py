import numpy as np
import vertex_screen as vt


# layout:  u (n_dim) f (1)  n_frames * [r (3) t(3) e (n_exp-1)]
def f_id_f(x, m_vertex, gt_landmarks, n_frames, n_id, f_blend_shape, w_reg):
    u = x[0:n_id]
    f = x[n_id]
    # p =  [  r (3)  t(3) e (dim_exp)   , n_frames  ]
    p = x[n_id + 1:].reshape((n_frames, -1))
    e = p[:, 6:]
    e = np.c_[1 - np.sum(e, axis=1), e]
    blend_shape = f_blend_shape(u)  # (dim_exp, n_v_flat)
    vertexes = e @ blend_shape  # (n_frame,n_v_flat)
    vertexes = vertexes + m_vertex
    # add w=1
    vertexes = vertexes.reshape((n_frames, -1, 3))
    vertexes = np.concatenate([vertexes, np.ones((vertexes.shape[0], vertexes.shape[1], 1))], axis=2)

    # build transform matrix
    m_r1 = vt.build_rotate_matrix(p[:, 0], 0, 1)
    m_r2 = vt.build_rotate_matrix(p[:, 1], 0, 2)
    m_r3 = vt.build_rotate_matrix(p[:, 2], 1, 2)
    m_t = vt.build_transition_matrix(p[:, 3:6])

    m_proj = vt.build_projection_matrix(f)

    mat = m_proj @ (m_t @ m_r1 @ m_r2 @ m_r3)
    vertexes = np.transpose(vertexes, (0, 2, 1))
    vertexes = mat @ vertexes
    vertexes = np.transpose(vertexes, (0, 2, 1))
    vertexes = vertexes[:, :, 0:2] / vertexes[:, :, 3][:, :, None]
    e_landmarks_1 = vertexes - gt_landmarks

    e_reg_1 = np.average(p, axis=0) - p
    return np.concatenate([e_landmarks_1.flatten(), e_reg_1.flatten() * w_reg, np.array((np.sum(u ** 2) - 1,))])


def build_initial_guess_and_bound(n_frames, u, n_exp):
    n_exp_1 = n_exp - 1
    f_guess = np.array([1.5], dtype=np.float32)
    rt_guess = np.array([0, 0, 0, 0, 0, -2], dtype=np.float32)
    e_guess = np.ones(n_exp_1, dtype=np.float32) / n_exp_1
    p_row = np.concatenate((rt_guess, e_guess))
    initial_guess = np.concatenate((u, f_guess, np.tile(p_row, n_frames)))
    p_lbound = np.concatenate(
        (np.array([-2, -2, -2, -np.inf, -np.inf, -np.inf], dtype=np.float32),
         np.full(e_guess.shape, 0, dtype=np.float32)))
    p_ubound = np.concatenate(
        (np.array([2, 2, 2, np.inf, np.inf, 0], dtype=np.float32), np.full(e_guess.shape, 1, dtype=np.float32)))
    lbound = np.concatenate((
        np.full(u.shape, -1, dtype=np.float32),
        np.full((1,), 0, dtype=np.float32),
        np.tile(p_lbound, n_frames)
    ))
    ubound = np.concatenate((
        np.full(u.shape, 1, dtype=np.float32),
        np.full((1,), np.inf, dtype=np.float32),
        np.tile(p_ubound, n_frames)
    ))
    return initial_guess, (lbound, ubound)


# x0 = 1  # initial guess
# optimize_result = op.least_squares(f_id_f, x0, verbose=2, args=(gt_landmarks, n_frames, f_blend_shape, w_reg))


def build_initial_guess_and_bound_p(n_frames, u, n_exp):
    n_exp_1 = n_exp - 1
    f_guess = np.array([1.5], dtype=np.float32)
    rt_guess = np.array([0, 0, 0, 0, 0, -2], dtype=np.float32)
    e_guess = np.ones(n_exp_1, dtype=np.float32) / n_exp_1
    p_row = np.concatenate((rt_guess, e_guess))
    initial_guess = np.concatenate((u, f_guess, np.tile(p_row, n_frames)))
    p_lbound = np.concatenate(
        (np.full(rt_guess.shape, -np.inf, dtype=np.float32), np.full(e_guess.shape, 0, dtype=np.float32)))
    p_ubound = np.concatenate(
        (np.full(rt_guess.shape, np.inf, dtype=np.float32), np.full(e_guess.shape, 1, dtype=np.float32)))
    lbound = np.concatenate((
        np.full(u.shape, -1, dtype=np.float32),
        np.full((1,), 0, dtype=np.float32),
        np.tile(p_lbound, n_frames)
    ))
    ubound = np.concatenate((
        np.full(u.shape, 1, dtype=np.float32),
        np.full((1,), np.inf, dtype=np.float32),
        np.tile(p_ubound, n_frames)
    ))
    return initial_guess, (lbound, ubound)


def f_p(x, m_vertex, gt_landmarks, f, blend_shape, w_reg):
    e = x[6:]
    e = np.concatenate([np.array(1 - np.sum(e))[None], e])
    vertexes = e @ blend_shape  # (n_frame,n_v_flat)
    vertexes = vertexes + m_vertex
    # add w=1
    # vertexes = vt.rt_multiframe(vertexes,x[None,0:6],f)
    vertexes = vertexes.reshape((-1, 3))
    vertexes = np.concatenate([vertexes, np.ones((vertexes.shape[0], 1))], axis=1)

    # build transform matrix
    m_r1 = vt.build_rotate_matrix(x[None, 0], 0, 1)
    m_r2 = vt.build_rotate_matrix(x[None, 1], 0, 2)
    m_r3 = vt.build_rotate_matrix(x[None, 2], 1, 2)
    m_t = vt.build_transition_matrix(x[None, 3:6])

    m_proj = vt.build_projection_matrix(f)

    mat = m_proj @ (m_t @ m_r1 @ m_r2 @ m_r3)[0]
    vertexes = np.transpose(vertexes)
    vertexes = mat @ vertexes
    vertexes = np.transpose(vertexes)
    vertexes = vertexes[:, 0:2] / vertexes[:, 3][:, None]

    e_landmarks_1 = vertexes - gt_landmarks
    return np.concatenate([e_landmarks_1.flatten(), e * w_reg])


def f_p_2(x, m_vertex, gt_landmarks, f, blend_shape, w_reg):
    e = x[6:]
    e = np.concatenate([np.array(1 - np.sum(e))[None], e])
    vertexes = e @ blend_shape  # (n_frame,n_v_flat)
    vertexes = vertexes + m_vertex
    # add w=1
    vertexes = vt.rt_multiframe(vertexes, x[None, 0:6], f)

    e_landmarks_1 = vertexes - gt_landmarks
    return np.concatenate([e_landmarks_1.flatten(), e * w_reg])


def f_p_f_2(x, m_vertex, gt_landmarks, blend_shape, w_reg):
    e = x[7:]
    f = x[6]
    e = np.concatenate([np.array(1 - np.sum(e))[None], e])
    vertexes = e @ blend_shape  # (n_frame,n_v_flat)
    vertexes = vertexes + m_vertex
    # add w=1
    vertexes = vt.rt_multiframe(vertexes, x[None, 0:6], f)

    e_landmarks_1 = vertexes - gt_landmarks
    return np.concatenate([e_landmarks_1.flatten(), e * w_reg])


def f_sq(x, fp, *args):
    return np.sum(fp(x, *args) ** 2)


def f_rt(x, m_vertex, gt_landmarks, f, blend_shape, w_reg):
    vertexes = blend_shape[7]  # (n_frame,n_v_flat)
    vertexes = vertexes + m_vertex
    # add w=1
    vertexes = vertexes.reshape((-1, 3))
    vertexes = np.concatenate([vertexes, np.ones((vertexes.shape[0], 1))], axis=1)

    # build transform matrix
    m_r1 = vt.build_rotate_matrix(x[None, 0], 0, 1)
    m_r2 = vt.build_rotate_matrix(x[None, 1], 0, 2)
    m_r3 = vt.build_rotate_matrix(x[None, 2], 1, 2)
    m_t = vt.build_transition_matrix(x[None, 3:6])

    m_proj = vt.build_projection_matrix(f)

    mat = m_proj @ (m_t @ m_r1 @ m_r2 @ m_r3)[0]
    vertexes = np.transpose(vertexes)
    vertexes = mat @ vertexes
    vertexes = np.transpose(vertexes)
    vertexes = vertexes[:, 0:2] / vertexes[:, 3][:, None]
    if vertexes.shape != gt_landmarks.shape:
        raise ValueError("shape mismatch")
    e_landmarks_1 = vertexes - gt_landmarks
    return np.concatenate([e_landmarks_1.flatten()])
