from pythongl import *
import threading
from PIL import Image
import numpy as np


def read_mesh_desc_file():
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


# TODO define a func to pass vertexes, rotation, projection matrix
global_parameters = read_mesh_desc_file()


def load_parameters():
    vertexes, triangles, landmarks = global_parameters
    return vertexes, triangles, landmarks


def draw_mesh(v, t, x):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glVertexPointer(3, GL_FLOAT, 0, v)
    # fill triangles
    glDrawElements(GL_TRIANGLES, t.size, GL_UNSIGNED_INT, t)

    # mesh lines
    glDisable(GL_LIGHTING)
    glColor4f(0.2, 0.2, 0.2, 0.5)
    glDrawElements(GL_LINE_LOOP, t.size, GL_UNSIGNED_INT, t)

    # show landmarks
    glDisable(GL_DEPTH_TEST)
    glColor3fv((1, 0, 0))
    glPointSize(8)
    glDrawElements(GL_POINTS, x.size, GL_UNSIGNED_INT, x)
    glDisableClientState(GL_VERTEX_ARRAY)


def draw2():
    vertexes, triangles, landmarks = load_parameters()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glLoadIdentity()
    gluPerspective(75, 800 / 800, 0.1, 87)
    gluLookAt(*(0.5, 0.6, 2), *(0, 0.1, 0), *(0, 1, 0))

    # 画人头
    glDisable(GL_TEXTURE_2D)
    draw_mesh(vertexes, triangles, landmarks)

    # 画叠上去
    glLoadIdentity()
    glEnable(GL_TEXTURE_2D)
    glColor4f(1, 1, 1, 0.5)
    with quads(texture=1):
        glVertex2f(-1, -1)
        glTexCoord2f(1, 0)
        glVertex2f(1, -1)
        glTexCoord2f(1, 1)
        glVertex2f(1, 1)
        glTexCoord2f(0, 1)
        glVertex2f(-1, 1)
        glTexCoord2f(0, 0)

    glutSwapBuffers()


def init():
    glutInit()
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)

    glutInitWindowSize(800, 800)
    window = glutCreateWindow(b"opengl")
    glClearColor(0, 0, 0, 0)
    glClearDepth(1.0)
    glEnable(GL_CULL_FACE)

    glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 1, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (1, 1, 1, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1, 1, 1, 1))
    glEnable(GL_LIGHT0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    glEnable(GL_POINT_SMOOTH)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glEnable(GL_TEXTURE_2D)
    glGenTextures(10)
    img = Image.open('test.png')
    width, height = img.size
    img = img.tobytes('raw', 'RGBA', 0, -1)
    glBindTexture(GL_TEXTURE_2D, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, 4,
                 width, height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, img)
    glTexParameterf(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexEnvf(GL_TEXTURE_ENV,
              GL_TEXTURE_ENV_MODE, GL_MODULATE)


def main_draw():
    init()
    glutDisplayFunc(draw2)
    glutMainLoop()


def nohub_draw_cycle():
    t = threading.Thread(target=main_draw)
    t.start()


if __name__ == '__main__':
    nohub_draw_cycle()
