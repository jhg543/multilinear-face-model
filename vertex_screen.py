import numpy as np


def rt(v, tx, ty, tz, r1, r2, r3):
    m_t = np.array([[1, 0, 0, tx],
                    [0, 1, 0, ty],
                    [0, 0, 1, tz],
                    [0, 0, 0, 1]], dtype=np.float32)
    m_r1 = np.array([[np.cos(r1), -np.sin(r1), 0, 0],
                     [np.sin(r1), np.cos(r1), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)
    m_r2 = np.array([[np.cos(r2), 0, -np.sin(r2), 0],
                     [0, 1, 0, 0],
                     [np.sin(r2), 0, np.cos(r2), 0],
                     [0, 0, 0, 1]], dtype=np.float32)
    m_r3 = np.array([[1, 0, 0, 0],
                     [0, np.cos(r3), -np.sin(r3), 0],
                     [0, np.sin(r3), np.cos(r3), 0],
                     [0, 0, 0, 1]], dtype=np.float32)

    m = m_t @ m_r1 @ m_r2 @ m_r3 @ v
    return m


def proj(v, f):
    m_proj = np.array([[f, 0, 0, 0],
                       [0, f, 0, 0],
                       [0, 0, -1, -0.01],
                       [0, 0, -1, 0]], dtype=np.float32)
    return m_proj @ v
