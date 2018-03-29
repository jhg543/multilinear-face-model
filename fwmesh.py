import numpy as np


def read_mesh_def():
    """
    reads mesh desc file from face warehouse obj file
    :return: (vertex(x1,y1,z1,x2,y2,z2),triangle(v11,v12,v13,v21,v22,v23))
    """
    vertexes = []
    triangles = []
    with open('face/shape_0.obj') as meshfile:
        for line in meshfile:
            row = line.split()
            if row[0] == 'v':
                vertexes += [float(i) for i in row[1:]]
            elif row[0] == 'f':
                v0, v1, v2, v3 = [int(s.split('/')[0]) - 1 for s in row[1:]]
                triangles += [v0, v1, v2, v0, v2, v3]
    landmark_vertex_index = [502, 683, 712, 742, 1339, 1748, 1838, 1913, 2163, 2167, 3175, 3194, 3208, 3218, 3226, 3227,
                             3239, 3280,
                             3558, 3648, 3859, 3875, 3901, 4217, 4242, 4256, 4267, 4273, 4333, 4343, 4354, 6084, 6089,
                             6112, 6116,
                             6140, 6328, 6456, 6634, 6698, 6703, 6743, 6748, 6801, 6883, 7080, 7101, 7148, 7163, 7173,
                             7191, 7225,
                             7248, 8799, 8814, 8815, 8817, 8865, 8935, 8952, 9342, 9415, 10314, 10459, 10538, 10655,
                             10778, 10855]
    return np.array(vertexes, np.float32), np.array(triangles, np.uint32), np.array(landmark_vertex_index, np.uint32)
