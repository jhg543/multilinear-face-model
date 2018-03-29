import fwmodel
import cv2
import tensor as T
import numpy as np
import draw
import fwmesh


def test():
    vertex, triangle, landmark = fwmesh.read_mesh_def()
    draw.write_parameters(0, vertex)
    draw.write_parameters(1, triangle)
    draw.write_parameters(2, landmark)
    draw.write_parameters(3, 1.5)
    draw.write_parameters(5, 0.8)
    draw.start_render_loop_thread()
    se, ue, si, ui, c, m = fwmodel.load_compact_svd('C:\\dev\\3dface\\svd2', 40, 47)
    c = T.mode_dot(c, ue, 1)  # we don't need SVD on exp axis

    def get_blend_shape(core_id, v_id):
        return T.mode_dot(core_id, np.expand_dims(v_id, axis=0), 0)[0]

    draw.write_parameters(0, get_blend_shape(c, ui[4])[0] + m)

    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if frame is None:
            raise Exception('摄像头坏了。')
        draw.write_parameters(4, frame)


if __name__ == '__main__':
    test()
