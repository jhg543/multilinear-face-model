import numpy as np
import vertex_screen as vt


# layout:  u (n_dim) f (1)  n_frames * [r (3) t(3) e (n_exp-1)]
def f_id(x, m_vertex, gt_landmarks, f, rte_guess, f_blend_shape, w_reg):
    u = x
    p = rte_guess
    e = p[:, 6:]
    e = np.c_[1 - np.sum(e, axis=1), e]
    blend_shape = f_blend_shape(u)  # (dim_exp, n_v_flat)
    vertexes = e @ blend_shape  # (n_frame,n_v_flat)
    vertexes = vertexes + m_vertex
    # add w=1
    vertexes = vt.rt_multiframe(vertexes, p[:, 0:6], f)
    e_landmarks_1 = vertexes - gt_landmarks

    return np.concatenate([e_landmarks_1.flatten(), w_reg * np.array((np.sum(u ** 2) - 1,))])


def f_p_2(x, m_vertex, gt_landmarks, f, blend_shape, w_reg):
    e = x[6:]
    e = np.concatenate([np.array(1 - np.sum(e))[None], e])
    vertexes = e @ blend_shape  # (n_frame,n_v_flat)
    vertexes = vertexes + m_vertex
    # add w=1
    vertexes = vt.rt_multiframe(vertexes, x[None, 0:6], f)

    e_landmarks_1 = vertexes - gt_landmarks
    return np.concatenate([e_landmarks_1.flatten(), e * w_reg])


def f_p_fp(x, m_vertex, gt_landmarks, f, blend_shape, w_reg, rp, w_regrp):
    ff = f_p_2(x, m_vertex, gt_landmarks, f, blend_shape, w_reg)
    reg_rp = rp - x
    return np.concatenate([ff, w_regrp * reg_rp])


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
