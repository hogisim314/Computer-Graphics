from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

azimuth = 0.
elevation = 5
distance = 5
w_vec = glm.vec3(0,0,1)
v_vec = glm.vec3(0,0,1)
u_vec = glm.vec3(0,0,1)
Vup_vec = glm.vec3(0,1,0)
targetpos = glm.vec3(0,0,0)
camerapos = glm.vec3(0,0,5)
cameradirection = glm.vec3(0,0,-1)
leftButtonPressed = False
rightButtonPressed = False
start_xpos=0
start_ypos=0
start_zpos=0
cur_xpos=0
cur_ypos=0
cur_zpos=0
V = 1
offsetvec = glm.vec3(0,0,0)
prev_offsetX=0
prev_offsetY=0
offsetX=0
offsetY=0
mode=True

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

def prepare_V_mat():
    global azimuth, elevation, cur_xpos, cur_ypos, cur_zpos, start_xpos, start_ypos, start_zpos
    global w_vec, u_vec, v_vec, Vup_vec
    global camerapos, targetpos, V, distance

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
    global g_cam_ang, g_cam_height, mode
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS :
            if key==GLFW_KEY_V:
                if mode == True :
                    mode = False
                else : 
                    mode = True
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

def prepare_vao_grid(i):
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
        -3.0, 0.0, i*0.1,  1.0, 1.0, 1.0, # x-axis start
        3.0, 0.0, i*0.1,   1.0, 1.0, 1.0, # x-axis end 
        -3.0, 0.0, -i*0.1,  1.0, 1.0, 1.0, # z-axis start
        3.0, 0.0, -i*0.1,   1.0, 1.0, 1.0, # z-axis end 
       i*0.1, 0.0, -3.0,  1.0, 1.0, 1.0, # x-axis start
        i*0.1, 0.0, 3.0,   1.0, 1.0, 1.0, # x-axis end 
        -i*0.1, 0.0, -3.0,  1.0, 1.0, 1.0, # z-axis start
        -i*0.1, 0.0, 3.0,   1.0, 1.0, 1.0, # z-axis end  
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


def main():
    global azimuth, elevation, cur_xpos, cur_ypos, cur_zpos, start_xpos, start_ypos, start_zpos, leftButtonPressed, rightButtonPressed
    global cameradirection, camerapos, targetpos, offsetX, offsetY, mode
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

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)
        
        vao_frame = prepare_vao_frame()
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES,0,6)

        for i in range(30) :
            vao_frame = prepare_vao_grid(i)
            glBindVertexArray(vao_frame)
            glDrawArrays(GL_LINES, 0, 8)
        
        if mode == True :
            P = glm.perspective(45, 1, 1, 100)
        else : 
            P = glm.ortho(-1,1,-1,1,-100,100)
        
        # poll events
        glfwPollEvents()
        prepare_V_mat()
        V = glm.lookAt(glm.vec3(camerapos), glm.vec3(targetpos),glm.vec3(Vup_vec))

        # projection matrix
        # use orthogonal projection (we'll see details later)
        I = glm.mat4()
        
        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    
        # draw current frame

        # swap front and back buffers
        glfwSwapBuffers(window)

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
