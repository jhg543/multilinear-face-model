#include <python.h>
#include <iostream>
#include <fstream>
using namespace std;

void draw_init(){
    PyRun_SimpleString("import draw");
}

//非阻塞的绘图循环
void draw_cycle(){
    PyRun_SimpleString("draw.nohub_draw_cycle()");
}

//输入模型数据，每次输入会更新画面，有延迟
//输入是相当于obj内容的字符串……
void draw_load_data(string s){
    PyRun_SimpleString(("draw.str_to_vertex_and_face('''"+s+"''')").c_str());
}

int main(int argc, char* argv[]){
    ifstream t("face/shape_0.obj");
    string str((std::istreambuf_iterator<char>(t)),
               std::istreambuf_iterator<char>());
    Py_Initialize();

    draw_init();
    draw_cycle();
    draw_load_data(str);

    Py_Finalize();
}