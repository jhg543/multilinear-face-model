import numpy as np
import vertex_screen as vt

dim_id = 40
dim_exp = 46


def f_id_f(x, gt_landmarks, n_frames, f_blend_shape, f_):
    u = x[0:dim_id]
    f = x[dim_id]
    # p =  [  r (3)  t(3) e (dim_exp)   , n_frames  ]
    p = x[dim_id + 1:].reshape((n_frames, -1))
    r = p[:, 0:3]
    t = p[:, 3:6]
    e = p[:, 6:]
    e = np.c_[e, 1 - np.sum(e, axis=1)]
    blend_shape = f_blend_shape(u)  # (dim_exp, n_v_flat)
    vertexes = e @ blend_shape  # (n_frame,n_v_flat)

    # add w=1
    vertexes = vertexes.reshape((n_frames, -1, 3))
    vertexes = np.concatenate([vertexes, np.ones((vertexes.shape[0], vertexes.shape[1], 1))], axis=2)

    # build transform matrix
    m_r1 = vt.build_rotate_matrix(p[:, 0], 0, 1)
    m_r2 = vt.build_rotate_matrix(p[:, 1], 0, 2)
    m_r3 = vt.build_rotate_matrix(p[:, 2], 1, 2)
    m_t = vt.build_transition_matrix(p[3:6])

    m_proj = vt.build_projection_matrix(f)

    mat = m_proj @ (m_t @ m_r1 @ m_r2 @ m_r3)
    vertexes = vertexes @ mat
    vertexes = vertexes[:, 0:2] / vertexes[:, 2][:, None]
    vertexes -= gt_landmarks
