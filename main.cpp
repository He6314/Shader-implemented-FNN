#define GLM_ENABLE_EXPERIMENTAL

#include <windows.h>

#include <GL/glew.h>

#include <GL/freeglut.h>

#include <GL/gl.h>
#include <GL/glext.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include "InitShader.h"
#include "LoadMesh.h"
#include "LoadTexture.h"
#include "imgui_impl_glut.h"

#define PI 3.141592653589793f

static const std::string mesh_name = "models/BLJ/BlackLeatherJacket.obj";
static const std::string texture_name = "models/BLJ/Main Texture/[Albedo].jpg";

float time_sec = 0.0f;

float angleY = 0.0f;
float angleX = 0.0f;
float angleZ = 0.0f;

float posY = 0.0f;
float posX = 0.0f;
float posZ = 0.0f;

const int NUM_BUFFERS = 2;
const int NUM_PASS = 2;
GLuint paraBuffers[NUM_BUFFERS] = { -1 };
GLuint fbo = -1;

GLuint shader_program[NUM_PASS] = { -1 };
GLuint texture_id = -1;
MeshData mesh_data;
static const std::string vertex_shader[NUM_PASS] = { "shaders/p1_vs.glsl", "shaders/p2_vs.glsl" };
static const std::string fragment_shader[NUM_PASS] = { "shaders/p1_fs.glsl", "shaders/p2_fs.glsl" };

void reload_shader();

void draw_gui()
{
   ImGui_ImplGlut_NewFrame();

   if (ImGui::Button("Reload Shader"))
   {
	   reload_shader();
   }

   //create a slider to change the angle variables
   ImGui::SliderFloat("View angle x", &angleX, -PI, +PI);
   ImGui::SliderFloat("View angle y", &angleY, -PI, +PI);
   ImGui::SliderFloat("View angle z", &angleZ, -PI, +PI);

   ImGui::SliderFloat("Model pos x", &posX, -2.f, +2.f);
   ImGui::SliderFloat("Model pos y", &posY, -2.f, +2.f);
   ImGui::SliderFloat("Model pos z", &posZ, -2.f, +2.f);

   ImGui::Image((void*)texture_id, ImVec2(128.f, 128.f), ImVec2(0.0, 1.0), ImVec2(1.0, 0.0));

   ImGui::SameLine();
   ImGui::Image((void*)paraBuffers[0], ImVec2(128.f, 128.f), ImVec2(0.0, 1.0), ImVec2(1.0, 0.0));
   ImGui::SameLine();
   ImGui::Image((void*)paraBuffers[1], ImVec2(128.f, 128.f), ImVec2(0.0, 1.0), ImVec2(1.0, 0.0));

   static bool show = false;
   ImGui::ShowTestWindow();
   ImGui::Render();
 }

void display()
{
   //clear the screen
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   const int w = glutGet(GLUT_WINDOW_WIDTH);
   const int h = glutGet(GLUT_WINDOW_HEIGHT);
   const float aspect_ratio = float(w) / float(h);

   glm::mat4 M = 
	   glm::rotate(angleX, glm::vec3(1.0f, 0.0f, 0.0f))*
	   glm::rotate(angleY, glm::vec3(0.0f, 1.0f, 0.0f))*
	   glm::rotate(angleZ, glm::vec3(0.0f, 0.0f, 1.0f))*
	   glm::translate(glm::vec3(posX,posY,posZ))*
	   glm::scale(glm::vec3(mesh_data.mScaleFactor));
   glm::mat4 V = glm::lookAt(glm::vec3(0.0f, 1.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
   glm::mat4 P = glm::perspective(3.141592f / 4.0f, aspect_ratio, 0.1f, 100.0f);
   glm::mat4 PVM = P*V*M;

/*/////////////////////////////////////
   PASS1
/////////////////////////////////////*/
   glUseProgram(shader_program[0]);

   glBindFramebuffer(GL_FRAMEBUFFER, fbo); // Render to FBO, all gbuffer textures
   const GLenum drawBuffers[NUM_BUFFERS] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
   glDrawBuffers(NUM_BUFFERS, drawBuffers);
   glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texture_id);
   int tex_loc = glGetUniformLocation(shader_program[0], "diffuse_color");
   glUniform1i(tex_loc, 0); // we bound our texture to texture unit 0
   int PVM_loc = glGetUniformLocation(shader_program[0], "PVM");
   glUniformMatrix4fv(PVM_loc, 1, false, glm::value_ptr(PVM));

   glBindVertexArray(mesh_data.mVao);
   glDrawElements(GL_TRIANGLES, mesh_data.mNumIndices, GL_UNSIGNED_INT, 0);

/*/////////////////////////////////////
   PASS2
/////////////////////////////////////*/
   glUseProgram(shader_program[1]);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDrawBuffer(GL_BACK);

   glClearColor(0.35f, 0.35f, 0.35f, 0.0f);
   glBindTexture(GL_TEXTURE_2D, texture_id);
   glUniform1i(tex_loc, 0);
   glUniformMatrix4fv(PVM_loc, 1, false, glm::value_ptr(PVM));
   
   glBindVertexArray(mesh_data.mVao);
   glDrawElements(GL_TRIANGLES, mesh_data.mNumIndices, GL_UNSIGNED_INT, 0);

   draw_gui();

   glutSwapBuffers();
}

void idle()
{
	glutPostRedisplay();

   const int time_ms = glutGet(GLUT_ELAPSED_TIME);
   time_sec = 0.001f*time_ms;
}

bool reload_shader_pass(int i) {
	bool reloadFlag = false;
	GLuint new_shader = InitShader(vertex_shader[i].c_str(), fragment_shader[i].c_str());
	if (new_shader == -1) // loading failed
	{
		reloadFlag = false;
	}
	else
	{
		if (shader_program[i] != -1)
		{
			glDeleteProgram(shader_program[i]);
		}
		shader_program[i] = new_shader;
		reloadFlag = true;
	}
	return reloadFlag;
}

void reload_shader()
{
	bool reloadFlag = true;

	for (int i = 0; i < NUM_PASS; i++) {
		reloadFlag *= reload_shader_pass(i);
	}
   if(!reloadFlag) // loading failed
   {
      glClearColor(1.0f, 0.0f, 1.0f, 0.0f);
   }
   else
   {
      glClearColor(0.35f, 0.35f, 0.35f, 0.0f);
      if(mesh_data.mVao != -1)
      {
         BufferIndexedVerts(mesh_data);
      }
   }
}

void printGlInfo()
{
	std::cout << "Template modified by Qin He" <<std::endl;
	std::cout << std::endl;
	std::cout << "OpenGL information===========================" << std::endl;
	std::cout << "Vendor: "       << glGetString(GL_VENDOR)                    << std::endl;
	std::cout << "Renderer: "     << glGetString(GL_RENDERER)                  << std::endl;
	std::cout << "Version: "      << glGetString(GL_VERSION)                   << std::endl;
	std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION)  << std::endl;
}

void initOpenGl()
{
   //Initialize glew so that new OpenGL function names can be used
   glewInit();

   glEnable(GL_DEPTH_TEST);

   reload_shader();

   //Load a mesh and a texture
   mesh_data = LoadMesh(mesh_name); //Helper function: Uses Open Asset Import library.
   texture_id = LoadTexture(texture_name.c_str()); //Helper function: Uses FreeImage library

   const int w = glutGet(GLUT_WINDOW_WIDTH);
   const int h = glutGet(GLUT_WINDOW_HEIGHT);

   glGenTextures(NUM_BUFFERS, paraBuffers);

   for (int i = 0; i<NUM_BUFFERS; i++)
   {
	   glBindTexture(GL_TEXTURE_2D, paraBuffers[i]);
	   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, 0);
	   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	   glBindTexture(GL_TEXTURE_2D, 0);
   }

   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   //attach the texture we just created to color attachment 1
   for (int i = 0; i<NUM_BUFFERS; i++)
   {
	   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, paraBuffers[i], 0);
   }
   //unbind the fbo
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// glut callbacks need to send keyboard and mouse events to imgui
void keyboard(unsigned char key, int x, int y)
{
   ImGui_ImplGlut_KeyCallback(key);
   std::cout << "key : " << key << ", x: " << x << ", y: " << y << std::endl;

   switch(key)
   {
      case 'r':
      case 'R':
         reload_shader();     
      break;
   }
}

void keyboard_up(unsigned char key, int x, int y)
{
   ImGui_ImplGlut_KeyUpCallback(key);
}

void special_up(int key, int x, int y)
{
   ImGui_ImplGlut_SpecialUpCallback(key);
}

void passive(int x, int y)
{
   ImGui_ImplGlut_PassiveMouseMotionCallback(x,y);
}

void special(int key, int x, int y)
{
   ImGui_ImplGlut_SpecialCallback(key);
}

void motion(int x, int y)
{
   ImGui_ImplGlut_MouseMotionCallback(x, y);
}

void mouse(int button, int state, int x, int y)
{
   ImGui_ImplGlut_MouseButtonCallback(button, state);
}


int main (int argc, char **argv)
{
   glutInit(&argc, argv); 
   glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
   glutInitWindowPosition (5, 5);
   glutInitWindowSize (1280, 720);
   int win = glutCreateWindow ("OpenGL Template");

   printGlInfo();

   glutDisplayFunc(display); 
   glutKeyboardFunc(keyboard);
   glutSpecialFunc(special);
   glutKeyboardUpFunc(keyboard_up);
   glutSpecialUpFunc(special_up);
   glutMouseFunc(mouse);
   glutMotionFunc(motion);
   glutPassiveMotionFunc(motion);

   glutIdleFunc(idle);

   initOpenGl();
   ImGui_ImplGlut_Init();

   glutMainLoop();

   glutDestroyWindow(win);
   return 0;		
}


