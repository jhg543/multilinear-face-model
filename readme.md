# opengl 畫臉

## 簡介

什麼畫臉打出來我就後悔了……   
其實是畫obj文件，我主要用這東西來畫臉所以就變成opengl畫臉了。 

運行起來大概長這樣: 
![樣例](樣例.jpg)

## 使用方法

運行draw.py來畫臉。 

main.cpp是c++綁定(不過感覺很隨便)。

## 依賴

python3

pip install pyopengl pillow

如果你要運行c++的話，還應當:

    -I "%python_path%/include"
    -L "%python_path%/libs"

Note:
If installing PyOpenGL from easy_install, pip or conda, make sure that you have already installed a GLUT implementation, such as FreeGLUT
