import numpy as np
import os
import tensor as T
import argparse


def read_one_blend_shape(file_path):
    with open(file_path, 'rb') as f:
        n_expression, n_vertex, n_face = np.fromfile(f, dtype=np.int32, count=3)  # it's constant 46,11510
        n_expression = n_expression + 1
        return np.fromfile(f, dtype=np.float32, count=n_expression * n_vertex * 3).reshape((n_expression, n_vertex * 3))


def read_blend_shapes(base_path):
    return np.stack(
        [read_one_blend_shape(os.path.join(base_path, '{0}_shape.bs'.format(i))) for i in range(1, 151)])


def calc_and_save_svd(tensor, base_path):
    mean = np.average(tensor, axis=(0, 1))
    mean.tofile(os.path.join(base_path, 'mean_vertex'))
    tensor = tensor - mean

    u_id, s_id, v = np.linalg.svd(T.unfold(tensor, 0), 0)
    s_id.tofile(os.path.join(base_path, 's_id_f32'))
    u_id.tofile(os.path.join(base_path, 'u_id_f32'))

    u_exp, s_exp, v = np.linalg.svd(T.unfold(tensor, 1), 0)
    s_exp.tofile(os.path.join(base_path, 's_exp_f32'))
    u_exp.tofile(os.path.join(base_path, 'u_exp_f32'))

    c0 = T.mode_dot(tensor, u_id.transpose(), 0)
    c1 = T.mode_dot(c0, u_exp.transpose(), 1)
    c1.tofile(os.path.join(base_path, 'core'))


def load_svd(base_path):
    s_exp = np.fromfile(os.path.join(base_path, 's_exp_f32'), dtype=np.float32, count=47)
    u_exp = np.fromfile(os.path.join(base_path, 'u_exp_f32'), dtype=np.float32, count=47 * 47).reshape((47, 47))
    s_id = np.fromfile(os.path.join(base_path, 's_id_f32'), dtype=np.float32, count=150)
    u_id = np.fromfile(os.path.join(base_path, 'u_id_f32'), dtype=np.float32, count=150 * 150).reshape((150, 150))
    core_tensor = np.fromfile(os.path.join(base_path, 'core'), dtype=np.float32, count=150 * 47 * 34530).reshape(
        (150, 47, 34530))
    mean_vertex = np.fromfile(os.path.join(base_path, 'mean_vertex'), dtype=np.float32, count=34530)
    return s_exp, u_exp, s_id, u_id, core_tensor, mean_vertex


def load_compact_svd(base_path, dim_id, dim_exp):
    s_exp, u_exp, s_id, u_id, core_tensor, mean_vertex = load_svd(base_path)
    return s_exp[0:dim_exp], u_exp[:, 0:dim_exp], s_id[0:dim_id], u_id[:, 0:dim_id], core_tensor[0:dim_id, 0:dim_exp,
                                                                                     :], mean_vertex


def restore_vector(core_tensor, v_id, v_exp):
    c0 = T.mode_dot(core_tensor, v_exp, 1)
    c1 = T.mode_dot(c0, v_id, 0)
    return c1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='perform HOSVD on mode id,exp of facewarehouse dataset')
    parser.add_argument('--datadir', help='path to directory containing dataset')
    parser.add_argument('--savedir', help='place to save result of SVD')
    args = parser.parse_args()
    calc_and_save_svd(read_blend_shapes(args.datadir), args.savedir)
