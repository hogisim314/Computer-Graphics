from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os
import time

azimuth = 0.
elevation = 5
distance = 5
w_vec = glm.vec3(0, 0, 1)
v_vec = glm.vec3(0, 0, 1)
u_vec = glm.vec3(0, 0, 1)
Vup_vec = glm.vec3(0, 1, 0)
targetpos = glm.vec3(0, 0, 0)
camerapos = glm.vec3(3, 2, 4)
cameradirection = glm.vec3(0, 0, -1)
leftButtonPressed = False
rightButtonPressed = False
start_xpos = 0
start_ypos = 0
start_zpos = 0
cur_xpos = 0
cur_ypos = 0
cur_zpos = 0
prev_offsetX = 0
prev_offsetY = 0
offsetX = 0
offsetY = 0
mode = True

channel_list = []
joint_list = []
motion_list = []
offset_list = []
parent_list = []
parent_index_list = []
name_stack = []
index_stack = []
channel_num_list = []
offset_vertex_list = []
npoffset_vertex_list = []
glmoffset_vertex_list = []
numOfJoint = 0
fps = 0
frames = 0
node_list = []
len = 0
dropped = 0
moving = 0
idx = 0
first = 0
frame_cnt=0
motionitr=0
color_loc=0
resize = 1
lineOrBox = 1

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

    w_vec = glm.vec3(np.cos(elevation)*np.sin(azimuth),
                     np.sin(elevation), np.cos(elevation)*np.cos(azimuth))
    u_vec = np.cross(Vup_vec, w_vec)
    u_vec /= np.sqrt(np.dot(u_vec, u_vec))
    v_vec = glm.vec3(np.cross(u_vec, w_vec))
    v_vec /= np.sqrt(np.dot(v_vec, v_vec))
    camerapos = glm.vec3(distance*w_vec.x+targetpos.x, distance *
                         w_vec.y+targetpos.y, distance*w_vec.z+targetpos.z)


def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------

    # vertex shader
    # create an empty shader object
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    # provide shader source code
    glShaderSource(vertex_shader, vertex_shader_source)
    # compile the shader object
    glCompileShader(vertex_shader)

    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())

    # fragment shader
    # create an empty shader object
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    # provide shader source code
    glShaderSource(fragment_shader, fragment_shader_source)
    # compile the shader object
    glCompileShader(fragment_shader)

    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    # create an empty program object
    shader_program = glCreateProgram()
    # attach the shader objects to the program object
    glAttachShader(shader_program, vertex_shader)
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


def key_callback(window, key, scancode, action, mods):
    global g_cam_ang, g_cam_height, mode, h_mode, wiremode, moving, lineOrBox
    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE)
            if key == GLFW_KEY_SPACE:
                moving = 1
            elif key == GLFW_KEY_1:
                lineOrBox = 1
            elif key == GLFW_KEY_2:
                lineOrBox = 2

def button_callback(window, button, action, mod):
    global leftButtonPressed, rightButtonPressed, start_xpos, start_ypos, cur_xpos, cur_ypos, azimuth, elevation
    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action == GLFW_PRESS:
            start_xpos, start_ypos = glfwGetCursorPos(window)
            cur_xpos, cur_ypos = glfwGetCursorPos(window)
            leftButtonPressed = True
        elif action == GLFW_RELEASE:
            leftButtonPressed = False
    if button == GLFW_MOUSE_BUTTON_RIGHT:
        if action == GLFW_PRESS:
            start_xpos, start_ypos = glfwGetCursorPos(window)
            cur_xpos, cur_ypos = glfwGetCursorPos(window)
            rightButtonPressed = True
        elif action == GLFW_RELEASE:
            rightButtonPressed = False


def cursor_callback(window, xpos, ypos):
    global cur_xpos, cur_ypos, leftButtonPressed, rightButtonPressed, camerapos
    global azimuth, elevation, Vup_vec, u_vec, w_vec, v_vec, targetpos, start_xpos, start_ypos

    if leftButtonPressed == True:
        start_xpos = cur_xpos
        start_ypos = cur_ypos
        cur_xpos, cur_ypos = glfwGetCursorPos(window)
        elevation -= (cur_ypos-start_ypos)/150
        if np.sin(elevation) < 0:
            azimuth -= (cur_xpos-start_xpos)/100
        else:
            azimuth += (cur_xpos-start_xpos)/100

    if rightButtonPressed == True:
        start_xpos = cur_xpos
        start_ypos = cur_ypos
        cur_xpos, cur_ypos = glfwGetCursorPos(window)
        targetpos += ((start_xpos-cur_xpos)*u_vec +
                      (start_ypos-cur_ypos)*v_vec)/500
        camerapos += ((start_xpos-cur_xpos)*u_vec +
                      (start_ypos-cur_ypos)*v_vec)/500


def scroll_callback(window, xoffset, yoffset):
    global distance
    distance *= np.power(1.1, float(-yoffset))


def drop_callback(window, paths):
    global joint_list, motion_list, channel_list, parent_list, name_stack, channel_num_list, numOfJoint, fps, frames, parent_index_list, len, dropped
    global npoffset_vertex_list, glmoffset_vertex_list, index_stack, offset_list, idx, first, moving,resize
    idx = 0
    dropped = 1
    joint_list = []
    motion_list = []
    channel_list = []
    parent_list = []
    name_stack = []
    index_stack = []
    channel_num_list = []
    numOfJoint = 0
    fps = 0
    frames = 0
    parent_index_list = []
    len = 0
    realPosition_list = []
    npoffset_vertex_list = []
    glmoffset_vertex_list = []
    offset_list = []
    first = 1
    moving=0
    resize = 1
    # print obj file name
    for path in paths:
        file_name = os.path.basename(path)
    bvhFile = open(path, 'r')
    lines = bvhFile.readlines()  # 파일 내용을 한 줄씩 읽어 리스트로 저장
    for line in lines:
        parts = line.split()
        if parts[0] == "ROOT":
            joint_list.append(parts[1])
            parent_list.append("None")
            parent_index_list.append(-1)
            name_stack.append(parts[1])
            index_stack.append(idx)
            numOfJoint += 1
            idx += 1
        elif parts[0] == "JOINT":
            joint_list.append(parts[1])
            parent_list.append(name_stack[-1])
            parent_index_list.append(index_stack[-1])
            numOfJoint += 1
            name_stack.append(parts[1])
            index_stack.append(idx)
            idx += 1
        elif parts[0] == "End":
            joint_list.append(parts[1])
            parent_list.append(name_stack[-1])
            parent_index_list.append(index_stack[-1])
            name_stack.append(parts[1])
            index_stack.append(idx)
            idx += 1
        elif parts[0] == "}":
            name_stack.pop()
            index_stack.pop()
        elif parts[0] == "OFFSET":
            x=float(parts[1])
            y=float(parts[2])
            z=float(parts[3])
            if (glm.dot(glm.vec3(x,y,z), glm.vec3(x,y,z))>1) :
                resize = 100
            offset_list.append(x/resize)
            offset_list.append(y/resize)
            offset_list.append(z/resize)
        elif parts[0] == "CHANNELS":
            channel_num_list.append(parts[1])
            for i in range(2, int(parts[1])+2):
                channel_list.append(parts[i])
        elif parts[0] == "Frames:":
            frames = int(parts[1])
        elif parts[0] == "Frame":
            fps = float(parts[2])
        elif parts[0] == "{" or parts[0] == "HIERARCHY" or parts[0] == "MOTION":
            continue
        else:
            for part in parts:
                motion_list.append(float(part))
    print("Obj file name : %s" % (file_name))
    print("Number of Frame : %d" % (frames))
    print("fps : %f" % (fps))
    print("Number of joints : %d" % (numOfJoint))
    print("List of all joint names :")
    for joint in joint_list:
        if (joint == "Site"):
            continue
        print(joint)
    idx -= 1

def prepare_node_list():
    global joint_list, motion_list, channel_list, parent_list, name_stack, channel_num_list, numOfJoint, fps, frames, parent_index_list, len, dropped
    global npoffset_vertex_list, glmoffset_vertex_list, index_stack, offset_list, node_list, idx
        # create a hirarchical model - Node(parent, link_transform_from_parent, shape_transform, color)
    for i in range(0, idx+1):
        if parent_list[i] == "None":
            node_list.append(Node(None, glm.mat4(), glm.mat4(), glm.vec3(1, 1, 1)))
        else:
            node_list.append(Node(node_list[parent_index_list[i]], glm.translate((offset_list[i*3+0], offset_list[i*3+1], offset_list[i*3+2])), glm.mat4(), glm.vec3(1, 1, 1)))
            


def prepare_vao_grid(i):
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
                         # position        # color
                         -10.0, 0.0, i*0.1,  1.0, 1.0, 1.0,  # x-axis start
                         10.0, 0.0, i*0.1,   1.0, 1.0, 1.0,  # x-axis end
                         -10.0, 0.0, -i*0.1,  1.0, 1.0, 1.0,  # z-axis start
                         10.0, 0.0, -i*0.1,   1.0, 1.0, 1.0,  # z-axis end
                         i*0.1, 0.0, -10.0,  1.0, 1.0, 1.0,  # x-axis start
                         i*0.1, 0.0, 10.0,   1.0, 1.0, 1.0,  # x-axis end
                         -i*0.1, 0.0, -10.0,  1.0, 1.0, 1.0,  # z-axis start
                         -i*0.1, 0.0, 10.0,   1.0, 1.0, 1.0,  # z-axis end
                         )

    # create and activate VAO (vertex array object)
    # create a vertex array object ID and store it to VAO variable
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    # create a buffer object ID and store it to VBO variable
    VBO = glGenBuffers(1)
    # activate VBO as a vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                 vertices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
                         # position        # color
                         -3.0, 0.0, 0.0,  1.0, 0.0, 0.0,  # x-axis start
                         3.0, 0.0, 0.0,  1.0, 0.0, 0.0,  # x-axis end
                         0.0, 0.0, 3.0,  0.0, 0.0, 1.0,  # y-axis start
                         0.0, 0.0, -3.0,  0.0, 0.0, 1.0,  # y-axis end
                         )

    # create and activate VAO (vertex array object)
    # create a vertex array object ID and store it to VAO variable
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    # create a buffer object ID and store it to VBO variable
    VBO = glGenBuffers(1)
    # activate VBO as a vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                 vertices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_bvh_line_modified(list):
    nptemp_list = np.array(list, dtype=np.float32)
    glmtemp_list = glm.array(nptemp_list, dtype=glm.float32)
    # prepare vertex data (in main memory)
    vertices = glmtemp_list
    # create and activate VAO (vertex array object)
    # create a vertex array object ID and store it to VAO variable
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    # create a buffer object ID and store it to VBO variable
    VBO = glGenBuffers(1)
    # activate VBO as a vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                 vertices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)
    return VAO

def prepare_vao_cube():
    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    vertices = glm.array(glm.float32,
        # position      normal
        -1 ,  1 ,  1 ,  0, 0, 1, # v0
         1 , -1 ,  1 ,  0, 0, 1, # v2
         1 ,  1 ,  1 ,  0, 0, 1, # v1

        -1 ,  1 ,  1 ,  0, 0, 1, # v0
        -1 , -1 ,  1 ,  0, 0, 1, # v3
         1 , -1 ,  1 ,  0, 0, 1, # v2

        -1 ,  1 , -1 ,  0, 0,-1, # v4
         1 ,  1 , -1 ,  0, 0,-1, # v5
         1 , -1 , -1 ,  0, 0,-1, # v6

        -1 ,  1 , -1 ,  0, 0,-1, # v4
         1 , -1 , -1 ,  0, 0,-1, # v6
        -1 , -1 , -1 ,  0, 0,-1, # v7

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 ,  1 ,  0, 1, 0, # v1
         1 ,  1 , -1 ,  0, 1, 0, # v5

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 , -1 ,  0, 1, 0, # v5
        -1 ,  1 , -1 ,  0, 1, 0, # v4
 
        -1 , -1 ,  1 ,  0,-1, 0, # v3
         1 , -1 , -1 ,  0,-1, 0, # v6
         1 , -1 ,  1 ,  0,-1, 0, # v2

        -1 , -1 ,  1 ,  0,-1, 0, # v3
        -1 , -1 , -1 ,  0,-1, 0, # v7
         1 , -1 , -1 ,  0,-1, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 , -1 ,  1 ,  1, 0, 0, # v2
         1 , -1 , -1 ,  1, 0, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 , -1 , -1 ,  1, 0, 0, # v6
         1 ,  1 , -1 ,  1, 0, 0, # v5

        -1 ,  1 ,  1 , -1, 0, 0, # v0
        -1 , -1 , -1 , -1, 0, 0, # v7
        -1 , -1 ,  1 , -1, 0, 0, # v3

        -1 ,  1 ,  1 , -1, 0, 0, # v0
        -1 ,  1 , -1 , -1, 0, 0, # v4
        -1 , -1 , -1 , -1, 0, 0, # v7
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

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_bvh_line():
    global offset_vertex_list, npoffset_vertex_list, glmoffset_vertex_list
    # prepare vertex data (in main memory)
    vertices = glmoffset_vertex_list
    # create and activate VAO (vertex array object)
    # create a vertex array object ID and store it to VAO variable
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    # create a buffer object ID and store it to VBO variable
    VBO = glGenBuffers(1)
    # activate VBO as a vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                 vertices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)
    return VAO


class Node:
    def __init__(self, parent, link_transform_from_parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.link_transform_from_parent = link_transform_from_parent
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_joint_transform(self, joint_transform):
        self.joint_transform *= joint_transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform_from_parent * self.joint_transform
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform

    def get_shape_transform(self):
        return self.shape_transform

    def get_color(self):
        return self.color
    
    def get_position(self):
        return self.get_global_transform() * glm.vec4(0,0,0,1).xyz
    
    def init_joint_transform(self):
        self.joint_transform = glm.mat4()
    
    def get_parent(self):
        return self.parent

def draw_node(vao, node, color_loc):
    # MVP = VP * node.get_global_transform() * node.get_shape_transform()
    color = node.get_color()

    glBindVertexArray(vao)
    # glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawArrays(GL_LINES, 0, 2)

def get_rotate_matrix(curPos, parentcurPos):
    offset = glm.vec3(curPos-parentcurPos)
    len = glm.length(offset)
    vec1 = glm.normalize(glm.vec3(0,len,0))
    vec2 = glm.normalize(offset)

    dot = glm.dot(vec1, vec2)
    cross = glm.cross(vec1, vec2)
    if cross==glm.vec3(0,0,0):
        return glm.mat4()
    axis = glm.normalize(cross)
    angle = glm.acos(dot)

    rotate = glm.rotate(angle,axis)
    return rotate

def draw_cube(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_TRIANGLES, 0, 36)

def draw_one_frame(vao_cube,P,V, MVP_loc):
    global joint_list, node_list, channel_num_list, channel_list, frame_cnt, motionitr,color_loc, resize
    if frame_cnt==frames or frame_cnt==0:
        frame_cnt=0
        motionitr=0
    for node in node_list :
        node.init_joint_transform()
    cnt = 0
    nodeitr = 0
    channel_numitr = 0
    channelitr = 0
    for _ in range(0,idx+1):# 노드 돌아가면서,,
        if (joint_list[nodeitr]=="Site"):
            nodeitr+=1
            continue
        node = node_list[nodeitr]
        channel_num = channel_num_list[channel_numitr]
        for _ in range(0, int(channel_num)):
            motion = motion_list[motionitr]
            channel = channel_list[channelitr]
            if channel == "XPOSITION" or channel == "Xposition":
                node.set_joint_transform(glm.translate((motion/resize, 0, 0)))
            elif channel == "YPOSITION" or channel == "Yposition":
                node.set_joint_transform(glm.translate((0, motion/resize, 0)))
            elif channel == "ZPOSITION" or channel == "Zposition":
                node.set_joint_transform(glm.translate((0, 0, motion/resize)))
            elif channel == "XROTATION" or channel == "Xrotation":
                node.set_joint_transform(glm.rotate(np.deg2rad(motion), (1, 0, 0)))
            elif channel == "YROTATION" or channel == "Yrotation":
                node.set_joint_transform(glm.rotate(np.deg2rad(motion), (0, 1, 0)))
            elif channel == "ZROTATION" or channel == "Zrotation":
                node.set_joint_transform(glm.rotate(np.deg2rad(motion), (0, 0, 1)))
            # print("frame_cnt is %d joint_list[nodeitr] is %s motion is %f channel is %s"%(frame_cnt, joint_list[nodeitr], motion, channel))
            motionitr += 1
            channelitr += 1
        cnt+=1
        nodeitr += 1
        channel_numitr += 1
    node_list[0].update_tree_global_transform()
    for node in node_list:
        if (node.get_parent()==None):
            continue
        templist = []
        curPos = node.get_position()
        parentcurPos = node.get_parent().get_position()
        templist.append(curPos.x)
        templist.append(curPos.y)
        templist.append(curPos.z)
        templist.append(0.0)
        templist.append(1.0)
        templist.append(0.0)
        templist.append(parentcurPos.x)
        templist.append(parentcurPos.y)
        templist.append(parentcurPos.z)
        templist.append(0.0)
        templist.append(1.0)
        templist.append(0.0)
        vao_bvh = prepare_vao_bvh_line_modified(templist)
        draw_node(vao_bvh,node,color_loc)
        if lineOrBox==2:
                len = glm.length(curPos-parentcurPos)
                rotateMat = get_rotate_matrix(curPos, parentcurPos)
                Matrix = glm.translate((curPos+parentcurPos)/2)*rotateMat*glm.scale((0.02,len/2, 0.02))
                draw_cube(vao_cube,P*V*Matrix, MVP_loc)
        
    frame_cnt+=1

def main():
    global azimuth, elevation, cur_xpos, cur_ypos, cur_zpos, start_xpos, start_ypos, start_zpos, leftButtonPressed, rightButtonPressed
    global cameradirection, camerapos, targetpos, offsetX, offsetY, mode, dropped, first, moving, node_list, channel_num_list, channel_list
    global motion_list, numOfJoint, fps, frames,color_loc, frame_cnt, lineOrBox

    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)  # for macOS

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

    vao_cube = prepare_vao_cube()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        glfwPollEvents()

        if mode == True:
            P = glm.perspective(45, 1, 1, 100)
        else:
            P = glm.ortho(-1, 1, -1, 1, -100, 100)
        prepare_V_mat()
        V = glm.lookAt(glm.vec3(camerapos), glm.vec3(targetpos), glm.vec3(Vup_vec))
        M = glm.mat4()
        MVP = P*V*M
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(view_pos_loc, camerapos.x, camerapos.y, camerapos.z)

        glUniform3f(color_loc, 1, 1, 1)
        vao_frame = prepare_vao_frame()
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 4)

        for i in range(100):
            vao_frame = prepare_vao_grid(i)
            glBindVertexArray(vao_frame)
            glDrawArrays(GL_LINES, 0, 8)
        if dropped == 1 and first == 1 and moving==0:  # 처음 드롭
            vao_bvh_list = []
            node_list = []
            prepare_node_list()
            node_list[0].update_tree_global_transform()
            for node in node_list:
                if node.get_parent()==None:
                    continue
                templist = []
                curPos = node.get_position()
                parentcurPos = node.get_parent().get_position()
                templist.append(curPos.x)
                templist.append(curPos.y)
                templist.append(curPos.z)
                templist.append(0.0)
                templist.append(1.0)
                templist.append(0.0)
                templist.append(parentcurPos.x)
                templist.append(parentcurPos.y)
                templist.append(parentcurPos.z)
                templist.append(0.0)
                templist.append(1.0)
                templist.append(0.0)
                vao_bvh = prepare_vao_bvh_line_modified(templist)
                vao_bvh_list.append(vao_bvh)
                draw_node(vao_bvh,node,color_loc)
            first = 0
            frame_cnt = 0
        if dropped == 1 and first == 0 and moving==0:  # 준비과정은 필요없음.
            i=0
            for node in node_list:
                if node.get_parent()==None:
                    continue
                curPos = node.get_position()
                parentcurPos = node.get_parent().get_position()
                if lineOrBox ==1:
                    draw_node(vao_bvh_list[i], node, color_loc)
                    i+=1
                if lineOrBox==2:
                    len = glm.length(curPos-parentcurPos)
                    rotateMat = get_rotate_matrix(curPos, parentcurPos)
                    Matrix = glm.translate((curPos+parentcurPos)/2)*rotateMat*glm.scale((0.02,len/2, 0.02))
                    draw_cube(vao_cube,P*V*Matrix, MVP_loc)

        if dropped ==1 and moving == 1:
            draw_one_frame(vao_cube,P,V,MVP_loc)
            time.sleep(fps)

        # swap front and back buffers
        glfwSwapBuffers(window)

    # terminate glfw
    glfwTerminate()


if __name__ == "__main__":
    main()

