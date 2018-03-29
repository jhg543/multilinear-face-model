import contextlib
import functools

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import os

@contextlib.contextmanager
def temp_translate(x,y,z):
    glTranslatef(x,y,z)
    yield
    glTranslatef(-x,-y,-z)

@contextlib.contextmanager
def temp_scale(x,y,z):
    glScalef(x,y,z)
    yield
    glScalef(1/x,1/y,1/z)


@contextlib.contextmanager
def new_list(list_id):
    glNewList(list_id,GL_COMPILE)
    yield
    glEndList()

@contextlib.contextmanager
def line_base(color=None,width=None,line_type=GL_LINES):
    if color:
        glColor3fv(color)
    if width:
        glLineWidth(width)
    glBegin(line_type)
    yield
    glEnd()
lines=functools.partial(line_base, line_type=GL_LINES)
line_loop=functools.partial(line_base, line_type=GL_LINE_LOOP)

@contextlib.contextmanager
def points(size=None,color=None):
    if color:
        glColor3fv(color)
    if size:
        glPointSize(size)
    glBegin(GL_POINTS)
    yield
    glEnd() 

@contextlib.contextmanager
def quads(texture=None):
    if texture:
        glBindTexture(GL_TEXTURE_2D, texture)
    glBegin(GL_QUADS)
    yield
    glEnd()
    if texture:
        glBindTexture(GL_TEXTURE_2D, 0)
    

