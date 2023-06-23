from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os
import re

azimuth = 0.
elevation = 5
distance = 5
w_vec = glm.vec3(0,0,1)
v_vec = glm.vec3(0,0,1)
u_vec = glm.vec3(0,0,1)
Vup_vec = glm.vec3(0,1,0)
targetpos = glm.vec3(0,0,0)
camerapos = glm.vec3(3,2,4)
cameradirection = glm.vec3(0,0,-1)
leftButtonPressed = False
rightButtonPressed = False
start_xpos=0
start_ypos=0
start_zpos=0
cur_xpos=0
cur_ypos=0
cur_zpos=0
offsetvec = glm.vec3(0,0,0)
prev_offsetX=0
prev_offsetY=0
offsetX=0
offsetY=0
mode=True
vertex = []
vertex_normal = []
faces = []
npvertex = np.array([])
npvertex_normal =np.array([])
npfaces= np.array([])
npvertex_info = np.array([])

glmvertex = []
glmvertex_normal = []
glmfaces =[]
glmvertex_info = []
vertexcnt=[]
tempnormal = []
vao_obj =0
flag = -1
h_mode = 0

glmvertex1=0
glmnormal1=0
glmfaces1=0
glmvertex_info1=0

glmvertex2=0
glmnormal2=0
glmfaces2=0
glmvertex_info2=0

glmvertex3=0
glmnormal3=0
glmfaces3=0
glmvertex_info3=0

glmvertex4=0
glmnormal4=0
glmfaces4=0
glmvertex_info4=0

wiremode = 0

faces_num=face3=face4=over4=0
tempnormal = []
vertex_info = []

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''

g_vertex_shader_src_for_obj = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP_obj;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP_obj * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 material_color;
void main()
{
    // light and material properties
    vec3 light_pos = vec3(3,2,4);
    vec3 light_pos1 = vec3(3,2,4);
    vec3 light_color = vec3(0,1,1);
    vec3 light_color1 = vec3(1,0,1);
    float material_shininess = 32.0;
    float material_shininess1 = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    vec3 light_ambient1 = 0.1*light_color1;
    vec3 light_diffuse1 = light_color1;
    vec3 light_specular1 = light_color1;


    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // for non-metal material

    vec3 material_ambient1 = material_color;
    vec3 material_diffuse1 = material_color;
    vec3 material_specular1 = light_color1;  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;
    vec3 ambient1 = light_ambient1 * material_ambient1;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    vec3 normal1 = normalize(vout_normal);
    vec3 surface_pos1 = vout_surface_pos;
    vec3 light_dir1 = normalize(light_pos1 - surface_pos1);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    float diff1 = max(dot(normal1, light_dir1), 0);
    vec3 diffuse1 = diff1 * light_diffuse1 * material_diffuse1;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 view_dir1 = normalize(view_pos - surface_pos1);
    vec3 reflect_dir1 = reflect(-light_dir1, normal1);
    float spec1 = pow( max(dot(view_dir1, reflect_dir1), 0.0), material_shininess1);
    vec3 specular1 = spec1 * light_specular1 * material_specular1;

    vec3 color = ambient + diffuse + specular;
    vec3 color1 = ambient1 + diffuse1 + specular1;

    FragColor = vec4((color+color1), 1.);
}
'''

def prepare_V_mat():
    global azimuth, elevation, cur_xpos, cur_ypos, cur_zpos, start_xpos, start_ypos, start_zpos
    global w_vec, u_vec, v_vec, Vup_vec
    global camerapos, targetpos, distance

    w_vec = glm.vec3(np.cos(elevation)*np.sin(azimuth), np.sin(elevation), np.cos(elevation)*np.cos(azimuth))
    u_vec = np.cross(Vup_vec,w_vec)
    u_vec /=np.sqrt(np.dot(u_vec,u_vec))
    v_vec = glm.vec3(np.cross(u_vec,w_vec))
    v_vec /=np.sqrt(np.dot(v_vec, v_vec))
    camerapos = glm.vec3(distance*w_vec.x+targetpos.x, distance*w_vec.y+targetpos.y,distance*w_vec.z+targetpos.z)
    
def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

# def key_callback(window, key, scancode, action, mods):
#     if action==GLFW_PRESS and key == GLFW_KEY_V:

def key_callback(window, key, scancode, action, mods):
    global g_cam_ang, g_cam_height, mode, h_mode, wiremode
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS :
            if key==GLFW_KEY_V:
                if mode == True :
                    mode = False
                else : 
                    mode = True
            if key==GLFW_KEY_H :
                if h_mode ==0 :
                    h_mode=1
                elif h_mode ==1 :
                    h_mode=0
                    
            if key==GLFW_KEY_Z :
                if wiremode==1 :
                    wiremode = 0
                else :
                    wiremode = 1
            if key == GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE)

def button_callback(window, button, action, mod):
    global leftButtonPressed, rightButtonPressed, start_xpos, start_ypos, cur_xpos, cur_ypos, azimuth, elevation
    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action==GLFW_PRESS:
            start_xpos, start_ypos = glfwGetCursorPos(window)
            cur_xpos, cur_ypos = glfwGetCursorPos(window)
            leftButtonPressed =True
        elif action==GLFW_RELEASE:
            leftButtonPressed=False 
    if button == GLFW_MOUSE_BUTTON_RIGHT:
        if action==GLFW_PRESS:
            start_xpos, start_ypos = glfwGetCursorPos(window)
            cur_xpos, cur_ypos = glfwGetCursorPos(window)
            rightButtonPressed=True
        elif action==GLFW_RELEASE:
            rightButtonPressed=False

def cursor_callback(window, xpos, ypos):
    global cur_xpos, cur_ypos, leftButtonPressed, rightButtonPressed, camerapos
    global azimuth, elevation, Vup_vec, u_vec,w_vec,v_vec, targetpos, start_xpos, start_ypos
    
    if leftButtonPressed==True:
        start_xpos = cur_xpos
        start_ypos = cur_ypos
        cur_xpos, cur_ypos = glfwGetCursorPos(window)
        elevation -=(cur_ypos-start_ypos)/150
        if np.sin(elevation) < 0 :
            azimuth -=(cur_xpos-start_xpos)/100
        else : 
            azimuth +=(cur_xpos-start_xpos)/100
        

    if rightButtonPressed==True:
        start_xpos=cur_xpos
        start_ypos = cur_ypos
        cur_xpos, cur_ypos = glfwGetCursorPos(window)
        targetpos+=((start_xpos-cur_xpos)*u_vec+(start_ypos-cur_ypos)*v_vec)/500
        camerapos+=((start_xpos-cur_xpos)*u_vec+(start_ypos-cur_ypos)*v_vec)/500

def scroll_callback(window, xoffset, yoffset):
    global distance 
    distance *= np.power(1.1, float(-yoffset))

def drop_callback(window, paths):
    global vertex, vertex_normal,faces, faces_num, face3, face4,over4, npvertex, npvertex_normal, npfaces, glmfaces, glmvertex, glmvertex_normal,glmvertex_info, vao_obj, flag
    global vertexcnt, tempnormal, vertex_info, h_mode
    vertex = []
    vertex_normal = []
    faces = []
    vertexcnt= []
    tempnormal=[]
    vertex_info= []
    flag = 1
    h_mode = 0
    #print obj file name
    for path in paths:
      file_name = os.path.basename(path)
      print("Obj file name : %s" %(file_name))

    objFile = open(path,'r')
    lines = objFile.readlines()  # 파일 내용을 한 줄씩 읽어 리스트로 저장
    faces_num=face3=face4=over4=0
    for line in lines:
        if line.startswith('v '):
            parts = line.split()
            vertex.append(float(parts[1]))
            vertex.append(float(parts[2]))
            vertex.append(float(parts[3]))
        elif line.startswith('vn '):
            parts = line.split()
            vertex_normal.append(float(parts[1]))
            vertex_normal.append(float(parts[2]))
            vertex_normal.append(float(parts[3]))
        elif line.startswith('f '):
            faces_num+=1
            num = re.findall(r'\d+', line)
            if (len(num)>6) :
                if (len(num)==8) :
                    face4+=1
                else:
                    over4+=1
                for i in range(2,len(num)-2,2) :
                    faces.append(int(num[0])-1)
                    faces.append(int(num[i])-1)
                    faces.append(int(num[i+2])-1)
            else:
                face3+=1
                faces.append(int(num[0])-1)
                faces.append(int(num[2])-1)
                faces.append(int(num[4])-1)
    numOfVertex = int(len(vertex) / 3)
    vertexcnt = np.array([0]*numOfVertex, dtype=np.int32)
    tempnormal = np.array([0.0]*numOfVertex*3, dtype = np.float32)
    for line in lines : 
        if line.startswith('f '):
            num = re.findall(r'\d+', line)
            for i in range(0,len(num),2):# int(num[i])-1해야 내가 원하는 index얻을 수 잇음
                vertexcnt[int(num[i])-1]+=1
                tempnormal[(int(num[i])-1)*3+0]+=vertex_normal[(int(num[i+1])-1)*3+0]
                tempnormal[(int(num[i])-1)*3+1]+=vertex_normal[(int(num[i+1])-1)*3+1]
                tempnormal[(int(num[i])-1)*3+2]+=vertex_normal[(int(num[i+1])-1)*3+2]
    
    for i in range (0,len(vertexcnt)) : 
        tempnormal[i*3+0] /=vertexcnt[i]
        tempnormal[i*3+1] /=vertexcnt[i]
        tempnormal[i*3+2] /=vertexcnt[i]
    
    for i in range(0, len(vertex), 3) :
        vertex_info.append(vertex[i])
        vertex_info.append(vertex[i+1])
        vertex_info.append(vertex[i+2])
        vertex_info.append(tempnormal[i])
        vertex_info.append(tempnormal[i+1])
        vertex_info.append(tempnormal[i+2])

    npfaces = np.array(faces, dtype=np.int32)
    npvertex_info = np.array(vertex_info, dtype = np.float32)

    glmfaces = glm.array(npfaces, dtype = glm.int32)
    glmvertex_info = glm.array(npvertex_info, dtype = glm.float32)
    # print(glmfaces)
    print ("Total number of faces : %d" %(faces_num))
    print ("Number of faces with 3 verties : %d" %(face3))
    print ("Number of faces with 4 verties : %d" %(face4))
    print ("Number of faces with more than 4 vertices : %d" %(over4))
    
def hierachical_render():
    global glmvertex_info1, glmnormal1, glmfaces1, glmvertex_info2, glmnormal2, glmfaces2, glmvertex_info3, glmnormal3, glmfaces3, glmvertex_info4, glmnormal4, glmfaces4
    mesh1_path = os.path.join(os.getcwd(),"obj_files","coral.obj")
    mesh2_path = os.path.join(os.getcwd(),"obj_files", "shark1.obj")
    mesh3_path = os.path.join(os.getcwd(),"obj_files", "starfish.obj")
    mesh4_path = os.path.join(os.getcwd(),"obj_files", "fish.obj")
    mesh1file = open(mesh1_path,'r')
    mesh2file = open(mesh2_path,'r')
    mesh3file = open(mesh3_path,'r')
    mesh4file = open(mesh4_path,'r')
    lines = mesh1file.readlines()
    glmvertex_info1, glmfaces1 = parsing_for_common_obj(lines)
    lines = mesh2file.readlines()
    glmvertex_info2, glmfaces2 = parsing_for_common_obj(lines)
    lines = mesh3file.readlines()
    glmvertex_info3, glmfaces3 = parsing_for_common_obj(lines)
    lines = mesh4file.readlines()
    glmvertex_info4, glmfaces4 = parsing_for_common_obj(lines)
    

def parsing_for_common_obj(lines) :
    vertex = []
    vertex_normal = []
    faces = []
    vertexcnt= []
    tempnormal=[]
    vertex_info=[]
    for line in lines:
        if line.startswith('v '):
            parts = line.split()
            vertex.append(float(parts[1]))
            vertex.append(float(parts[2]))
            vertex.append(float(parts[3]))
        elif line.startswith('vn '):
            parts = line.split()
            vertex_normal.append(float(parts[1]))
            vertex_normal.append(float(parts[2]))
            vertex_normal.append(float(parts[3]))
        elif line.startswith('f '):
            num = re.findall(r'\d+', line)
            if (len(num)>9) :
                for i in range(3,len(num)-3,3) :
                    faces.append(int(num[0])-1)
                    faces.append(int(num[i])-1)
                    faces.append(int(num[i+3])-1)
            else:
                faces.append(int(num[0])-1)
                faces.append(int(num[3])-1)
                faces.append(int(num[6])-1)
    numOfVertex = int(len(vertex) / 3)
    vertexcnt = np.array([0]*numOfVertex, dtype=np.int32)
    tempnormal = np.array([0.0]*numOfVertex*3, dtype = np.float32)
    for line in lines : 
        if line.startswith('f '):
            num = re.findall(r'\d+', line)
            for i in range(0,len(num), 3) :
                vertexcnt[int(num[i])-1]+=1
                tempnormal[(int(num[i])-1)*3+0]+=vertex_normal[(int(num[i+2])-1)*3+0]
                tempnormal[(int(num[i])-1)*3+1]+=vertex_normal[(int(num[i+2])-1)*3+1]
                tempnormal[(int(num[i])-1)*3+2]+=vertex_normal[(int(num[i+2])-1)*3+2]

    for i in range (0,len(vertexcnt)) : 
        tempnormal[i*3+0] /=vertexcnt[i]
        tempnormal[i*3+1] /=vertexcnt[i]
        tempnormal[i*3+2] /=vertexcnt[i]
    
    for i in range(0, len(vertex), 3) :
        vertex_info.append(vertex[i])
        vertex_info.append(vertex[i+1])
        vertex_info.append(vertex[i+2])
        vertex_info.append(tempnormal[i])
        vertex_info.append(tempnormal[i+1])
        vertex_info.append(tempnormal[i+2])

    npfaces = np.array(faces, dtype=np.int32)
    npvertex_info = np.array(vertex_info, dtype = np.float32)

    glmfaces = glm.array(npfaces, dtype = glm.int32)
    glmvertex_info = glm.array(npvertex_info, dtype = glm.float32)
    return glmvertex_info, glmfaces

def prepare_vao_grid(i):
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
        -10.0, 0.0, i*0.1,  1.0, 1.0, 1.0, # x-axis start
        10.0, 0.0, i*0.1,   1.0, 1.0, 1.0, # x-axis end 
        -10.0, 0.0, -i*0.1,  1.0, 1.0, 1.0, # z-axis start
        10.0, 0.0, -i*0.1,   1.0, 1.0, 1.0, # z-axis end 
       i*0.1, 0.0, -10.0,  1.0, 1.0, 1.0, # x-axis start
        i*0.1, 0.0, 10.0,   1.0, 1.0, 1.0, # x-axis end 
        -i*0.1, 0.0, -10.0,  1.0, 1.0, 1.0, # z-axis start
        -i*0.1, 0.0, 10.0,   1.0, 1.0, 1.0, # z-axis end  
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         -3.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         3.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, -3.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, 3.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, -3.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 3.0,  0.0, 0.0, 1.0, # z-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_obj(glmvertex_info, glmfaces):

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # create and activate EBO (element buffer object)
    EBO = glGenBuffers(1)   # create a buffer object ID and store it to EBO variable
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)  # activate EBO as an element buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, glmvertex_info.nbytes, glmvertex_info.ptr, GL_STATIC_DRAW) 

    # copy index data to EBO
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, glmfaces.nbytes, glmfaces.ptr, GL_STATIC_DRAW) 

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

class Node:
    def __init__(self, parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        #여기서부터 DFS 시작
        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color    
    
def draw_node(vao, node, VP, MVP_loc, color_loc, len):
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawElements(GL_TRIANGLES, len, GL_UNSIGNED_INT, None)

def main():
    global azimuth, elevation, cur_xpos, cur_ypos, cur_zpos, start_xpos, start_ypos, start_zpos, leftButtonPressed, rightButtonPressed, h_mode
    global cameradirection, camerapos, targetpos, offsetX, offsetY, mode, faces_num, flag, glmvertex_info, glmfaces
    global glmvertex_info1, glmfaces1, glmvertex_info2, glmfaces2, glmvertex_info3, glmfaces3, glmvertex_info4, glmfaces4
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, 'assignment1-2019073181', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetKeyCallback(window, key_callback)
    glfwSetDropCallback(window, drop_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)
    
    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    color_loc = glGetUniformLocation(shader_program, 'material_color')
    M_loc = glGetUniformLocation(shader_program, 'M')
    view_pos_loc = glGetUniformLocation(shader_program, 'view_pos')
    
    # shader_program_obj = load_shaders(g_vertex_shader_src_for_obj, g_fragment_shader_src)
    hierachical_render()

    # MVP_obj_loc = glGetUniformLocation(shader_program_obj,'MVP_obj')
    # color_obj_loc = glGetUniformLocation(shader_program_obj, 'material_color')
    # M_obj_loc = glGetUniformLocation(shader_program_obj, 'M')
    # view_pos_obj_loc = glGetUniformLocation(shader_program_obj, 'view_pos')

    coral = Node(None, glm.rotate(-np.pi/2,glm.vec3(1,0,0)), glm.vec3(0,.7,0))
    shark1 = Node(coral, glm.rotate(-np.pi/2,glm.vec3(1,0,0))*glm.rotate(np.pi, glm.vec3(0,0,1))*glm.translate((1.4, 1.4,4)), glm.vec3(0,1,1))
    shark2 = Node(coral, glm.rotate(-np.pi/2,glm.vec3(1,0,0))*glm.rotate(np.pi/3, glm.vec3(0,0,1))*glm.translate((-1.4, 3, 4)), glm.vec3(1,0,1))
    shark3 = Node(coral, glm.rotate(-np.pi/2,glm.vec3(1,0,0))*glm.translate((3,-1,4)) , glm.vec3(1,1,0))
    starfish1 = Node(coral, glm.rotate(-np.pi/2, glm.vec3(1,0,0))*glm.translate((2,0,0)), glm.vec3(0,1,1))
    starfish2 = Node(coral, glm.rotate(-np.pi/2, glm.vec3(1,0,0))*glm.translate((-2,0,0)), glm.vec3(0,1,1))
    starfish3 = Node(coral, glm.rotate(-np.pi/2, glm.vec3(1,0,0))*glm.translate((0,2,0)), glm.vec3(0,1,1))
    starfish4 = Node(coral, glm.rotate(-np.pi/2, glm.vec3(1,0,0))*glm.translate((0,-2,0)), glm.vec3(0,1,1))
    fish1= Node(shark1,glm.rotate(-np.pi/2, glm.vec3(1,0,0))*glm.translate((1.4,1.4,4))*glm.scale((.3,.3,.3)), glm.vec3(0,1,1))
    fish2= Node(shark1,glm.rotate(-np.pi/2, glm.vec3(1,0,0))*glm.translate((1.4,1.3,4))*glm.scale((.3,.3,.3)), glm.vec3(0,1,1))
    fish3= Node(shark2,glm.rotate(-np.pi/2, glm.vec3(1,0,0))*glm.translate((-1.6,2.8,4))*glm.scale((.3,.3,.3)), glm.vec3(1,0,1))
    fish4= Node(shark2,glm.rotate(-np.pi/2, glm.vec3(1,0,0))*glm.translate((-1.2,3.2,4))*glm.scale((.3,.3,.3)), glm.vec3(1,0,1))
    fish5= Node(shark3,glm.rotate(-np.pi/2, glm.vec3(1,0,0))*glm.translate((3.1,-0.7,4))*glm.scale((.3,.3,.3)), glm.vec3(1,1,0))
    fish6= Node(shark3,glm.rotate(-np.pi/2, glm.vec3(1,0,0))*glm.translate((3.3,-1.3,4))*glm.scale((.3,.3,.3)), glm.vec3(1,1,0))
    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        glfwPollEvents()         

        if mode == True :
            P = glm.perspective(45, 1, 1, 100)
        else : 
            P = glm.ortho(-1,1,-1,1,-100,100)
        prepare_V_mat()
        V = glm.lookAt(glm.vec3(camerapos), glm.vec3(targetpos),glm.vec3(Vup_vec))
        M = glm.mat4()  
        MVP = P*V*M
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(view_pos_loc, camerapos.x, camerapos.y, camerapos.z)

        glUniform3f(color_loc, 1,1,1)
        vao_frame = prepare_vao_frame()
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES,0,6)

        for i in range(100) :
            vao_frame = prepare_vao_grid(i)
            glBindVertexArray(vao_frame)
            glDrawArrays(GL_LINES, 0, 8)

        if (flag==1):
            vao_obj = prepare_vao_obj(glmvertex_info, glmfaces)
            glBindVertexArray(vao_obj)
            glDrawElements(GL_TRIANGLES,len(glmfaces),GL_UNSIGNED_INT, None)
        if (h_mode ==1):
            # glUseProgram(shader_program_obj)
            flag=0
            t = glfwGetTime()
            th = np.radians(t*90)
            coral.set_transform(glm.rotate(th/4,glm.vec3(0,1,0)))

            shark1.set_transform(glm.translate((3*glm.sin(t),0,0)))
            shark2.set_transform(glm.translate((-2*glm.sin(t),0,0)))
            shark3.set_transform(glm.translate((2*glm.sin(t),0,0)))

            starfish1.set_transform(glm.rotate(th/2,glm.vec3(0,1,0)))
            starfish2.set_transform(glm.rotate(th/2,glm.vec3(0,1,0)))
            starfish3.set_transform(glm.rotate(th/2,glm.vec3(0,1,0)))
            starfish4.set_transform(glm.rotate(th/2,glm.vec3(0,1,0)))

            fish1.set_transform(glm.rotate(th/4, glm.vec3(0,1,0)))
            fish2.set_transform(glm.rotate(th/4, glm.vec3(0,-1,0)))
            fish3.set_transform(glm.rotate(th/4, glm.vec3(0,1,0)))
            fish4.set_transform(glm.rotate(th/4, glm.vec3(0,-1,0)))
            fish5.set_transform(glm.rotate(th/4, glm.vec3(0,1,0)))
            fish6.set_transform(glm.rotate(th/4, glm.vec3(0,-1,0)))
            
            coral.update_tree_global_transform()
            
            coral_vao = prepare_vao_obj(glmvertex_info1*.1, glmfaces1)
            shark_vao = prepare_vao_obj(glmvertex_info2*.01, glmfaces2)
            starfish_vao = prepare_vao_obj(glmvertex_info3*.1, glmfaces3)
            fish_vao = prepare_vao_obj(glmvertex_info4*.1, glmfaces4)
            draw_node(coral_vao, coral, P*V, MVP_loc, color_loc, len(glmfaces1))
            draw_node(shark_vao, shark1, P*V, MVP_loc, color_loc, len(glmfaces2))
            draw_node(shark_vao, shark2, P*V, MVP_loc, color_loc, len(glmfaces2))
            draw_node(shark_vao, shark3, P*V, MVP_loc, color_loc, len(glmfaces2))
            draw_node(starfish_vao, starfish1, P*V, MVP_loc, color_loc, len(glmfaces3))
            draw_node(starfish_vao, starfish2, P*V, MVP_loc, color_loc, len(glmfaces3))
            draw_node(starfish_vao, starfish3, P*V, MVP_loc, color_loc, len(glmfaces3))
            draw_node(starfish_vao, starfish4, P*V, MVP_loc, color_loc, len(glmfaces3))
            draw_node(fish_vao, fish1, P*V, MVP_loc, color_loc, len(glmfaces4))
            draw_node(fish_vao, fish2, P*V, MVP_loc, color_loc, len(glmfaces4))
            draw_node(fish_vao, fish3, P*V, MVP_loc, color_loc, len(glmfaces4))
            draw_node(fish_vao, fish4, P*V, MVP_loc, color_loc, len(glmfaces4))
            draw_node(fish_vao, fish5, P*V, MVP_loc, color_loc, len(glmfaces4))
            draw_node(fish_vao, fish6, P*V, MVP_loc, color_loc, len(glmfaces4))


        if wiremode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        

        # swap front and back buffers
        glfwSwapBuffers(window)

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
