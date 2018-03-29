from pythongl import *
import threading
import numpy as np
import glfw

# 0 = vertex coordinate
# 1 = triangle index
# 2 = landmark index
# 3 = f in projection matrix
# 4 = 2d image to show
# 5 = ground truth opacity ( setting )
parameters = [None for i in range(6)]


def write_parameters(index, value):
    parameters[index] = value


def load_parameters():
    return parameters


def draw_mesh(vertexes, triangles, landmarks):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glVertexPointer(3, GL_FLOAT, 0, vertexes)
    # fill triangles
    glDrawElements(GL_TRIANGLES, triangles.size, GL_UNSIGNED_INT, triangles)

    # mesh lines
    glDisable(GL_LIGHTING)
    glColor4f(0.2, 0.2, 0.2, 0.5)
    glDrawElements(GL_LINE_LOOP, triangles.size, GL_UNSIGNED_INT, triangles)

    # show landmarks
    glDisable(GL_DEPTH_TEST)
    glColor3fv((1, 0, 0))
    glPointSize(8)
    glDrawElements(GL_POINTS, landmarks.size, GL_UNSIGNED_INT, landmarks)
    glDisableClientState(GL_VERTEX_ARRAY)


def render():
    vertexes, triangles, landmarks, f, gt, gt_opacity = load_parameters()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glLoadIdentity()
    glMultMatrixf(np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, -1, -1], [0, 0, -0.01, 0]], dtype=np.float32))
    # gluPerspective(75, 800 / 800, 0.1, 100)
    gluLookAt(*(0.5, 0.6, 2), *(0, 0.1, 0), *(0, 1, 0))

    # 画人头
    glDisable(GL_TEXTURE_2D)
    draw_mesh(vertexes, triangles, landmarks)

    # 画叠上去
    if gt is not None:
        set_photo(gt)
        glLoadIdentity()
        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, gt_opacity)
        with quads(texture=1):
            glVertex2f(-1, -1)
            glTexCoord2f(0, 1)
            glVertex2f(1, -1)
            glTexCoord2f(0, 0)
            glVertex2f(1, 1)
            glTexCoord2f(1, 0)
            glVertex2f(-1, 1)
            glTexCoord2f(1, 1)


def set_photo(img):
    height, width, channels = img.shape
    glBindTexture(GL_TEXTURE_2D, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, 4,
                 width, height, 0, GL_BGR,
                 GL_UNSIGNED_BYTE, img)
    glTexParameterf(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER, GL_LINEAR)


def init():
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
    glTexEnvf(GL_TEXTURE_ENV,
              GL_TEXTURE_ENV_MODE, GL_MODULATE)


def glut_loop():
    render()
    glutSwapBuffers()


def glut_flush_display():
    glutPostRedisplay()


def glut_main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 800)
    window = glutCreateWindow(b"opengl")
    init()
    glutDisplayFunc(glut_loop)
    glutIdleFunc(glut_flush_display)
    glutMainLoop()


def glfw_main():
    if not glfw.init():
        return
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(800, 600, "Display", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)
    init()
    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Render here, e.g. using pyOpenGL
        render()
        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    glfw.terminate()


def start_render_loop_thread():
    t = threading.Thread(target=glfw_main)
    t.start()


if __name__ == '__main__':
    import fwmesh

    vertex, triangle, landmark = fwmesh.read_mesh_def()
    write_parameters(0, vertex)
    write_parameters(1, triangle)
    write_parameters(2, landmark)
    write_parameters(3, 1.5)
    write_parameters(5, 0.8)
    start_render_loop_thread()
