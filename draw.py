from pythongl import *
import threading
import time
from PIL import Image

all_vertex=[]
red_vertex=[502, 683, 712, 742, 1339, 1748, 1838, 1913, 2163, 2167, 3175, 3194, 3208, 3218, 3226, 3227, 3239, 3280, 3558, 3648, 3859, 3875, 3901, 4217, 4242, 4256, 4267, 4273, 4333, 4343, 4354, 6084, 6089, 6112, 6116, 6140, 6328, 6456, 6634, 6698, 6703, 6743, 6748, 6801, 6883, 7080, 7101, 7148, 7163, 7173, 7191, 7225, 7248, 8799, 8814, 8815, 8817, 8865, 8935, 8952, 9342, 9415, 10314, 10459, 10538, 10655, 10778, 10855]
all_face=[]
need_reload_flag=False

def generator_to_vertex_and_face(generator):
    global all_vertex,all_face
    all_vertex=[]
    all_face=[]
    def v(x,y,z):
        x,y,z=[float(i) for i in (x,y,z)]
        all_vertex.append((x,y,z))
    def vt(x,y):
        x,y=[float(i) for i in (x,y)]
    def f(*p_list):
        p_list = [int(s.split('/')[0]) for s in p_list]
        all_face.append(p_list)
    
    for info in generator:
        eval('%s(*%s)'% (info[0],info[1:]))

    global need_reload_flag
    need_reload_flag=True

def str_to_vertex_and_face(string):
    def obj_gen():
        for line in string.split('\n'):
            if line:
                yield line.split()
    return generator_to_vertex_and_face(obj_gen())

def make_gllist():
    with new_list(99):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        for face in all_face:
            with quads():
                for vertex_number in face:
                    glVertex3fv(all_vertex[vertex_number-1])

        glDisable(GL_LIGHTING)
        glColor4f(0.2,0.2,0.2,0.5)
        for face in all_face:
            with line_loop():
                for vertex_number in face:
                    glVertex3fv(all_vertex[vertex_number-1])

        #画标记点
        glDisable(GL_DEPTH_TEST)
        for i,vertex in enumerate(all_vertex):
            if i in red_vertex:
                with points(size=8,color=(1,0,0)):
                    glVertex3fv(vertex)


def draw():
    global need_reload_flag
    if need_reload_flag:
        make_gllist()
        need_reload_flag=False
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glLoadIdentity()
    gluPerspective(75, 800/800, 0.1, 87)
    gluLookAt(*(0.5,0.6,2), *(0,0.1,0), *(0,1,0))

    #画人头
    glDisable(GL_TEXTURE_2D)
    glCallList(99)

    #画叠上去
    glLoadIdentity()
    glEnable(GL_TEXTURE_2D)
    glColor4f(1,1,1,0.5)
    with quads(texture=1):
        glVertex2f(-1,-1)
        glTexCoord2f(1,0)
        glVertex2f(1,-1) 
        glTexCoord2f(1,1)
        glVertex2f(1,1) 
        glTexCoord2f(0,1)
        glVertex2f(-1,1)
        glTexCoord2f(0,0)

    glutSwapBuffers()

def init():
    glutInit()
    glutInitDisplayMode(GLUT_RGB|GLUT_DOUBLE|GLUT_DEPTH)

    glutInitWindowSize(800,800)
    window = glutCreateWindow(b"opengl")
    glClearColor(0,0,0,0)
    glClearDepth(1.0)
    glEnable(GL_CULL_FACE)
    
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 1, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  (1, 1, 1, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (1, 1, 1, 1))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1, 1, 1, 1))
    glEnable(GL_LIGHT0)
    
    glMatrixMode (GL_PROJECTION)
    glLoadIdentity()
    
    glEnable(GL_POINT_SMOOTH)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT,GL_NICEST)
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
                GL_UNSIGNED_BYTE,img)
    glTexParameterf(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexEnvf(GL_TEXTURE_ENV,
                    GL_TEXTURE_ENV_MODE, GL_MODULATE)


def main_draw():
    init()
    glutDisplayFunc(draw)
    glutMainLoop()

def nohub_draw_cycle():
    t = threading.Thread(target=main_draw)
    t.start()

if __name__=='__main__':
    nohub_draw_cycle()

    with open('face/shape_0.obj') as obj:
        str_to_vertex_and_face(obj.read())