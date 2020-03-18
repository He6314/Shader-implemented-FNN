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
#include <iomanip>
#include <vector>

#include "InitShader.h"
#include "LoadMeshTangents.h"
#include "LoadTexture.h"
#include "imgui_impl_glut.h"
#include "TransMatrices.h"

#include "fcn.hpp"
#include "NetworkMatrices.hpp"

#define PI 3.141592653589793f

static const std::string mesh_name = "models/BLJ/BlackLeatherJacket.obj";
//static const std::string mesh_name = "models/sphere.obj";
static const std::string texture_name = "models/BLJ/Main Texture/[Albedo].jpg";

float time_sec = 0.0f;

float angleX = 0.0f;
float angleY = -0.7f;//0.0f;
float angleZ = 0.0f;

float posY = 0.4f;//0.0f;
float posX = 0.4f;//0.0f;
float posZ = 0.7f;//0.0f;

const int NUM_BUFFERS = 4;
const int NUM_PASS = 3;

const int MAX_PIXEL = 1000000;
const int INPUT_DIM = 6;
const int OUTPUT_DIM =  3;//1;//

GLuint paraBuffers[NUM_BUFFERS] = { -1 };
GLuint fbo = -1;
GLuint depth_rbo;

//GLuint cs_test = -1;

int num_sample = 0;

GLfloat bufferValue[NUM_BUFFERS][4];
float bufferX[MAX_PIXEL][INPUT_DIM+1] = { 0 }; // 3 for normal, 3 for view, 3 for light direction, 2 for texCoord, 1 for index;
float bufferY[MAX_PIXEL][OUTPUT_DIM+1] = { 0 }; // RGBA;

float** trainX;
float** trainY;
float** validX;
float** validY;

GLuint shader_program[NUM_PASS] = { -1 };
GLuint texture_id = -1;

GLuint foward_shader = -1;


TransUBO transUBO;
MeshData mesh_data;
GLuint quad_vao;

static const std::string vertex_shader[NUM_PASS] = { "shaders/Phong_vs.glsl", "shaders/network_vs.glsl" , "shaders/testQuad_vs.glsl" };
static const std::string fragment_shader[NUM_PASS] = { "shaders/Phong_fs.glsl", "shaders/network_fs.glsl",  "shaders/testQuad_fs.glsl" };

int wIn = INPUT_DIM;
int wOut = OUTPUT_DIM;
int numHiddenLayer = 8; //6;// 12;//5; //
int wHidden = 8; // 6;//5;//  4; // 12; // 10;//8;//
float alpha = 0.00003f;// 0.00025f;the first that works

int quadIn = 2;
int quadOut = 3;
int quadNoHiddenLayer = 4;
int quadWHidden = 8;

float maxVectorX[INPUT_DIM] = { 0 };
float minVectorX[INPUT_DIM] = { 0 };
float maxVectorY[OUTPUT_DIM] = { 0 };
float minVectorY[OUTPUT_DIM] = { 0 };

float aveVectorX[INPUT_DIM] = { 0 };
float aveVectorY[OUTPUT_DIM] = { 0 };

Fcn network(wIn, wOut);
FcnSSBO fcnMat(wIn, wOut, wHidden, numHiddenLayer);
AveVectorUBO aves(aveVectorX, INPUT_DIM, aveVectorY, OUTPUT_DIM);
//DataSSBO fcnData(INPUT_DIM, OUTPUT_DIM);


Fcn quadNetwork(2, 3);
FcnSSBO quadMat(2, 3, 8, 8);

char* fname = new char[50];

int nbEpoch = 0;
int nbBatch = 0;
bool trainFlag = FALSE;
float beta1 = 0.5f;
float beta2 = 0.5f;//distfunc
//alpha
int batchSize = 32;

bool testQuad = FALSE;

int interactLayer = 0;
int interactWeight1 = 0;
int interactWeight2 = 0;
int interactBias = 0;
bool ctrlBias = FALSE;
bool debugData = FALSE;
int debugDepthInd = 0;
int debugWidthInd = 0;
int debugBiasInd = 0;

bool randomArt = FALSE;

void reload_shader();

void draw_gui()
{
	ImGui_ImplGlut_NewFrame();

	//ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::Begin("Basic options");
	if (ImGui::Button("Reload Shader"))
	{
		reload_shader();
	}
	ImGui::SliderFloat("View angle x", &angleX, -PI, +PI);
	ImGui::SliderFloat("View angle y", &angleY, -PI, +PI);
	ImGui::SliderFloat("View angle z", &angleZ, -PI, +PI);
	ImGui::SliderFloat("Model pos x", &posX, -2.f, +2.f);
	ImGui::SliderFloat("Model pos y", &posY, -2.f, +2.f);
	ImGui::SliderFloat("Model pos z", &posZ, -2.f, +2.f);

	static int outL = numHiddenLayer + 1;
	ImGui::SliderInt("OutLayer", &outL, 0, numHiddenLayer + 1);
	glUniform1i(99, outL);

	ImGui::Image((void*)texture_id, ImVec2(128.f, 128.f), ImVec2(0.0, 1.0), ImVec2(1.0, 0.0));


	ImGui::InputText("Filename", &fname[0], 50);
	ImGui::SameLine();
	if (ImGui::Button("Load"))
	{
		fcnMat.ReadFromFile(string(fname));
	}
	ImGui::End();

	ImGui::Begin("Training Paras");
	ImGui::SliderInt("Batch Size", &batchSize, 1, 128);
	ImGui::SliderFloat("Beta1", &beta1, 0.1f, 1.f- 1e-5,  "%.5f", 2.5f);
	ImGui::SliderFloat("Beta2", &beta2, 0.1f, 1.f- 1e-5, "%.5f", 2.5f);
	ImGui::SliderFloat("Alpha", &alpha, 1e-8, 1e-2, "%.5f", 2.5f);
	ImGui::Checkbox("Simple Quad", &testQuad);
	ImGui::End();

	if (!testQuad) {
		ImGui::Begin("Handles");
		ImGui::SliderInt("Layer", &interactLayer, 0, fcnMat.Depth()-1);
		ImGui::SliderInt("From", &interactWeight1, 0, fcnMat.Width(interactLayer) - 1);
		ImGui::SliderInt("To", &interactWeight2, 0, fcnMat.Width(interactLayer + 1) - 1);
		ImGui::Checkbox("Ctrl Bias", &ctrlBias); ImGui::SameLine();
		ImGui::SliderInt("Bias", &interactBias, 0, fcnMat.Width(interactLayer + 1) - 1);
		int interLoc = 0;
		if (!ctrlBias) interLoc = fcnMat.Index(0, interactLayer, interactWeight1, interactWeight2);
		else interLoc = fcnMat.Index(1, interactLayer, interactBias);

		float* value = fcnMat.Mats(interLoc);
		ImGui::SliderFloat("Value", value, -.95f, .95f);

		ImGui::Checkbox("Debug", &debugData);
		if (debugData) {
			glUniform1i(91, 1);
			glUniform1i(92, interactLayer);
			glUniform1i(93, interactWeight1);
			glUniform1i(94, interactWeight2);
			glUniform1i(95, interactBias);
		}
		else glUniform1i(91, 0);

		ImGui::End();
	}
	else {
		ImGui::Begin("Handles");
		ImGui::SliderInt("Layer", &interactLayer, 0, quadMat.Depth()-1);
		ImGui::SliderInt("From", &interactWeight1, 0, quadMat.Width(interactLayer) - 1);
		ImGui::SliderInt("To", &interactWeight2, 0, quadMat.Width(interactLayer + 1) - 1);
		ImGui::Checkbox("Ctrl Bias", &ctrlBias); ImGui::SameLine();
		ImGui::SliderInt("Bias", &interactBias, 0, quadMat.Width(interactLayer + 1) - 1);
		int interLoc = 0;
		if (!ctrlBias) interLoc = quadMat.Index(0, interactLayer, interactWeight1, interactWeight2);
		else interLoc = quadMat.Index(1, interactLayer, interactBias);
		float* value = quadMat.Mats(interLoc);
		ImGui::SliderFloat("Value", value, -.95f, .95f);
		ImGui::End();
	}

	//ImGui::SetNextWindowPos(ImVec2(875, 0));
	ImGui::Begin("Buffers");
	for (int i = 0; i < NUM_BUFFERS; i++) {
		ImGui::Image((void*)paraBuffers[i], ImVec2(200.f, 200.f), ImVec2(0.0, 1.0), ImVec2(1.0, 0.0));
		ImGui::NewLine();
	}

	//ImGui::Image((void*)cs_test, ImVec2(32.f, 32.f), ImVec2(0.0, 1.0), ImVec2(1.0, 0.0));
   ImGui::End();

   //static bool show = false;
   //ImGui::ShowTestWindow();
   ImGui::Render();
 }

void display()
{
   //clear the screen
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   const int w = glutGet(GLUT_WINDOW_WIDTH);
   const int h = glutGet(GLUT_WINDOW_HEIGHT);
   const float aspect_ratio = float(w) / float(h);


   //一令一动-----------------------------------------------------
   if (trainFlag && nbEpoch < 800) {
	//------------------------------------------------------------

	//-------------------------------------------------------------
	//CPU   
	bool completeEpoch = network.TrainCPU_1batch(nbBatch, nbEpoch, beta1, beta2, alpha);
	  
	 //GPU
	  //fcnData.UpdateBatch(nbBatch);
	  //fcnData.PassDataToShader();
	  //bool completeEpoch = network.TrainShader_1batch(nbBatch, nbEpoch, beta1, beta2, alpha);
	 
	  //fcnMat.printMat(2);

	   //glBindBuffer(GL_SHADER_STORAGE_BUFFER, fcnData.idSSBO);
	   //GLfloat *ptr;
	   //ptr = (GLfloat *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE); //GL_READ_WRITE?
	   //std::cerr << ptr[1] << ", y=" << ptr[2] << ", z=" << ptr[3] << endl;
	   //glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	   //glBindBuffer(GL_SHADER_STORAGE_BUFFER, fcnData.idSSBO);
	   //GLfloat *ptr;
	   //ptr = (GLfloat *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE); //GL_READ_WRITE?
	   //std::cerr << nbBatch<< ", LOSS: " << ptr[2] << endl;
	   //glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	   //cerr << nbBatch << endl;
	   
	   
	   nbBatch++;
	   if (completeEpoch) {
		   nbEpoch++;
		   nbBatch = 0;
	   }
	   if (nbEpoch >= 800) {
		   nbEpoch = 0;
		   fcnMat.WriteToFile();
		   trainFlag = false;
	   }
   }

   fcnMat.PassDataToShader();
   //fcnData.PassDataToShader();
   //=============================================================

  // glUseProgram(foward_shader);
   //glBindImageTexture(7, cs_test, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);

  // glDispatchCompute(32, 32, 1);
  // glMemoryBarrier(GL_ALL_BARRIER_BITS);
   if (!testQuad)
   {
	   transUBO.mats.M =
		   glm::rotate(angleX, glm::vec3(1.0f, 0.0f, 0.0f))*
		   glm::rotate(angleY, glm::vec3(0.0f, 1.0f, 0.0f))*
		   glm::rotate(angleZ, glm::vec3(0.0f, 0.0f, 1.0f))*
		   glm::translate(glm::vec3(posX, posY - 1.0f, posZ))*
		   glm::scale(glm::vec3(mesh_data.mScaleFactor));
	   transUBO.mats.V = glm::lookAt(glm::vec3(0.0f, 1.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	   transUBO.mats.P = glm::perspective(3.141592f / 4.0f, aspect_ratio, 0.1f, 100.0f);
	   glm::mat4 PVM = transUBO.mats.P*transUBO.mats.V*transUBO.mats.M;

	 /*/////////////////////////////////////
	  PASS CS
   ``/////////////////////////////////////*/

	   /*/////////////////////////////////////
		  PASS1: traditional lighting models
	   /////////////////////////////////////*/
	   glUseProgram(shader_program[0]);

	   //DRAW ON SCREEN:
	   //glBindFramebuffer(GL_FRAMEBUFFER, 0);
	   //glDrawBuffer(GL_BACK);

	   //DRAW IN BUFFERS:
	   glClearColor(0.35f, 0.35f, 0.35f, 0.0f);
	   glBindFramebuffer(GL_FRAMEBUFFER, fbo); // Render to FBO, all gbuffer textures
	   const GLenum drawBuffers[NUM_BUFFERS] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
	   glDrawBuffers(NUM_BUFFERS, drawBuffers);
	   glClearColor(0.3f, 0.5f, 0.5f, 0.0f);
	   //glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	   glActiveTexture(GL_TEXTURE0);
	   glBindTexture(GL_TEXTURE_2D, texture_id);
	   int tex_loc = glGetUniformLocation(shader_program[0], "diffuse_color");
	   glUniform1i(tex_loc, 0); // we bound our texture to texture unit 0  
	   transUBO.PassDataToShader();

	   glBindVertexArray(mesh_data.mVao);
	   mesh_data.DrawMesh();
	   //glDrawElements(GL_TRIANGLES, mesh_data.mNumIndices, GL_UNSIGNED_INT, 0);

	/*/////////////////////////////////////
	   PASS2: FCNN
	/////////////////////////////////////*/
//=========================================================
	   //GLuint64 startTime, stopTime;
	   //unsigned int queryID[2];
	   //glGenQueries(2, queryID);
		  // glQueryCounter(queryID[0], GL_TIMESTAMP);
//========================================================

	   glUseProgram(shader_program[1]);
	   glBindFramebuffer(GL_FRAMEBUFFER, 0);
	   glDrawBuffer(GL_BACK);

	   glClearColor(0.35f, 0.35f, 0.35f, 0.0f);

	   //for (int i = 0; i < NUM_BUFFERS; i++) {
		  // glActiveTexture(GL_TEXTURE0);
		  // int bufferLoc = 11 + i;
		  // glBindTexture(GL_TEXTURE_2D, paraBuffers[i]);
		  // glUniform1i(bufferLoc, i);
	   //}

	   //------------TEMP---------------------
	   glActiveTexture(GL_TEXTURE10);
	   glBindTexture(GL_TEXTURE_2D, texture_id);
	   tex_loc = glGetUniformLocation(shader_program[1], "ambTexture");
	   glUniform1i(tex_loc, 10);
	   //-------------------------------------

	   glActiveTexture(GL_TEXTURE0);
	   glBindTexture(GL_TEXTURE_2D, paraBuffers[0]);
	   glUniform1i(10, 0);

	   glActiveTexture(GL_TEXTURE1);
	   glBindTexture(GL_TEXTURE_2D, paraBuffers[1]);
	   glUniform1i(11, 1);

	   glActiveTexture(GL_TEXTURE2);
	   glBindTexture(GL_TEXTURE_2D, paraBuffers[2]);
	   glUniform1i(12, 2);

	   glActiveTexture(GL_TEXTURE3);
	   glBindTexture(GL_TEXTURE_2D, paraBuffers[3]);
	   glUniform1i(13, 3);

	   transUBO.PassDataToShader();
	   aves.PassDataToShader(aveVectorX, aveVectorY);

	   glDepthMask(GL_FALSE);
	   glBindVertexArray(quad_vao);
	   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	   glBindVertexArray(0);
	   glDepthMask(GL_TRUE);

//===========================================================================================
	   //glQueryCounter(queryID[1], GL_TIMESTAMP);
	   //GLint stopTimerAvailable = 0;
	   //while (!stopTimerAvailable) {
		  // glGetQueryObjectiv(queryID[1],
			 //  GL_QUERY_RESULT_AVAILABLE,
			 //  &stopTimerAvailable);
	   //}

	   //glGetQueryObjectui64v(queryID[0], GL_QUERY_RESULT, &startTime);
	   //glGetQueryObjectui64v(queryID[1], GL_QUERY_RESULT, &stopTime);
	   //printf("Time spent on the GPU: %f ms\n", (stopTime - startTime) / 1000000.0);
//===========================================================================================
   }
   
   else {
	   glUseProgram(shader_program[2]);
	   glBindFramebuffer(GL_FRAMEBUFFER, 0);
	   glDrawBuffer(GL_BACK);
	   glClearColor(0.35f, 0.35f, 0.35f, 0.0f);

	   quadMat.PassCtrlToShader();
	   quadMat.PassDataToShader();

	   glDepthMask(GL_FALSE);
	   glBindVertexArray(quad_vao);
	   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	   glBindVertexArray(0);
	   glDepthMask(GL_TRUE);
   }

   draw_gui();
   glutSwapBuffers();


}

float* arrangeBuffer(int wIn, int wOut, int wHidden, int numHiddenLayer) {
	int weightSize = fcnMat.wSize; //wIn*wHidden + wOut*wHidden + (numHiddenLayer-1)*wHidden*wHidden;
	int biasSize = fcnMat.bSize;// wOut + numHiddenLayer*wHidden;
	int paraSize = weightSize + biasSize;

	float* loc = new float[paraSize];
	return loc;
}

void CPUtest(int trainSize, int validSize) {
	network.SetTrainData(trainX, trainY, trainSize, batchSize);
	network.SetValidData(validX, validY, validSize);

	trainFlag = true;

	//network.TrainCPU();
	//fcnMat.PassCtrlToShader();
	//fcnMat.PassDataToShader();
	//fcnMat.WriteToFile();
}

void splitData(int n) {
	int numTrain = n*0.8;
	int numValid = n - numTrain;

	trainX = new float*[numTrain];
	trainY = new float*[numTrain];
	validX = new float*[numValid];
	validY = new float*[numValid];

	std::vector<int> index;
	for(int i = 0; i<n; i++) index.push_back(i);
	std::random_shuffle(index.begin(), index.end());

	//float maxTrain = 0;
	int testN = 0;
	for (int i = 0; i < numTrain; i++) {
		int k = index[i];
		trainX[i] = new float[INPUT_DIM];
		trainY[i] = new float[OUTPUT_DIM];
		for (int j = 0; j < INPUT_DIM; j++) {
			trainX[i][j] = bufferX[k][j] -aveVectorX[j];
		}
		for (int j = 0; j < OUTPUT_DIM; j++) {
			trainY[i][j] = bufferY[k][j] -aveVectorY[j];
		}
		testN++;
	}
	std::cout << "num train: " << testN << std::endl;
//	std::cout << "max train:" << maxTrain << std::endl;

	testN = 0;
	//float maxTest;
	for (int i = 0; i < numValid; i++) {
		int k = index[i + numTrain];
		validX[i] = new float[INPUT_DIM];
		validY[i] = new float[OUTPUT_DIM];
		for (int j = 0; j < INPUT_DIM; j++) {
			validX[i][j] = bufferX[k][j] -aveVectorX[j];
		}
		for (int j = 0; j < OUTPUT_DIM; j++) {
			validY[i][j] = bufferY[k][j] -aveVectorY[j];
		//if (bufferY[k][j] > maxTest)maxTest = bufferY[k][j];
		}
		testN++;
	}
	std::cout << "num validation: " << testN << std::endl;
	//std::cout << "max test:" << maxTest << std::endl;

	CPUtest(numTrain, numValid);
}

void loadBuffer() {
	const int buffer_width = glutGet(GLUT_WINDOW_WIDTH);
	const int buffer_height = glutGet(GLUT_WINDOW_HEIGHT);

		for (int i = 0; i < buffer_width; i++)
			for (int j = 0; j < buffer_height; j++)
			{
				float isFore[4]{ 0 };
				glBindFramebuffer(GL_FRAMEBUFFER, fbo);
				glReadBuffer(GL_COLOR_ATTACHMENT1);
				glPixelStorei(GL_PACK_ALIGNMENT, 1);
				glReadPixels(i, j, 1, 1, GL_RGBA, GL_FLOAT, isFore);

				if (isFore[3] == 1.0) {
					bufferX[num_sample][0] = isFore[0];
					bufferX[num_sample][1] = isFore[1];
					bufferX[num_sample][2] = isFore[2];//normal

					float temp[4];
					glReadBuffer(GL_COLOR_ATTACHMENT2);
					glPixelStorei(GL_PACK_ALIGNMENT, 2);
					glReadPixels(i, j, 1, 1, GL_RGBA, GL_FLOAT, temp);
					bufferX[num_sample][3] = temp[0];
					bufferX[num_sample][4] = temp[1];
					bufferX[num_sample][5] = temp[2];//view
					//bufferX[num_sample][9] = temp[3];//texCoord.x
					//bufferX[num_sample][0] = temp[3];//texCoord.y;

					//glReadBuffer(GL_COLOR_ATTACHMENT3);
					//glPixelStorei(GL_PACK_ALIGNMENT, 3);
					//glReadPixels(i, j, 1, 1, GL_RGBA, GL_FLOAT, temp);
					//bufferX[num_sample][6] = temp[0];
					//bufferX[num_sample][7] = temp[1];
					//bufferX[num_sample][8] = temp[2];//light
					//bufferX[num_sample][10] = temp[3];//texCoord.y
					//bufferX[num_sample][1] = temp[3];//texCoord.y;

					//bufferX[num_sample][11] = n;
					bufferX[num_sample][6] = num_sample;
					//bufferX[num_sample][2] = n;

					for (int i = 0; i < INPUT_DIM; i++) {
						if (bufferX[num_sample][i] > maxVectorX[i]) maxVectorX[i] = bufferX[num_sample][i];
						if (bufferX[num_sample][i] < minVectorX[i]) minVectorX[i] = bufferX[num_sample][i];
					}

					glReadBuffer(GL_COLOR_ATTACHMENT0);
					glPixelStorei(GL_PACK_ALIGNMENT, 0);
					glReadPixels(i, j, 1, 1, GL_RGBA, GL_FLOAT, temp);
					bufferY[num_sample][0] = temp[0];
					bufferY[num_sample][1] = temp[1];
					bufferY[num_sample][2] = temp[2];
					bufferY[num_sample][3] = temp[3];

					for (int i = 0; i < OUTPUT_DIM; i++) {
						if (bufferY[num_sample][i] > maxVectorY[i]) maxVectorY[i] = bufferY[num_sample][i];
						if (bufferY[num_sample][i] < minVectorY[i]) minVectorY[i] = bufferY[num_sample][i];
					}

					//fcnData.EnterData(bufferX[num_sample], bufferY[num_sample], i, j);
					
					num_sample++;
				}
				
				for (int i = 0; i < INPUT_DIM; i++) {
					aveVectorX[i] = (maxVectorX[i] + minVectorX[i]) / 2;
					fcnMat.inputAve[i] = aveVectorX[i];
				}
				for (int i = 0; i < OUTPUT_DIM; i++) {
					aveVectorY[i] = (maxVectorY[i] + minVectorY[i]) / 2;
					fcnMat.outputAve[i] = aveVectorY[i];
				}
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
			}

		//fcnData.BatchSize(batchSize);
		//fcnData.SplitData();

	std::cout << "Number of samples now: "<< num_sample << std::endl;
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
		reloadFlag = reloadFlag&&reload_shader_pass(i);
	}


	//GLuint new_shader = InitShader("shaders/propogation_cs.glsl");
	//if (foward_shader != -1)
	//{
	//	glDeleteProgram(foward_shader);
	//}
	//foward_shader = new_shader;

   if(!reloadFlag) // loading failed
   {
      glClearColor(1.0f, 0.0f, 1.0f, 0.0f);
   }
   else
   {
      glClearColor(0.35f, 0.35f, 0.35f, 0.0f);
      //if(mesh_data.mVao != -1)
      //{
      //   BufferIndexedVerts(mesh_data);
      //}
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
   mesh_data = LoadMesh(mesh_name); //Helper function: Uses Open Asset Import library
   texture_id = LoadTexture(texture_name.c_str()); //Helper function: Uses FreeImage library

   transUBO.InitBuffer();

   const int w = glutGet(GLUT_WINDOW_WIDTH);
   const int h = glutGet(GLUT_WINDOW_HEIGHT);

   glGenVertexArrays(1, &quad_vao);

//BUFFERS(Textures)==========================================================================================
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

   glGenTextures(1, &depth_rbo);
   glBindTexture(GL_TEXTURE_2D, depth_rbo);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
   glBindTexture(GL_TEXTURE_2D, 0);

   //glGenTextures(1, &cs_test);
   //glBindTexture(GL_TEXTURE_2D, cs_test);
   //glTexStorage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 32, 32);
   //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   //glBindTexture(GL_TEXTURE_2D, 0);

   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   //attach the texture we just created to color attachment 1
   for (int i = 0; i<NUM_BUFFERS; i++)
   {
	   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, paraBuffers[i], 0);
   } 
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_rbo, 0);
   //unbind the fbo
   glBindFramebuffer(GL_FRAMEBUFFER, 0);

//NETWORK==========================================================================================
   fcnMat.InitBuffer();
   fcnMat.InitData();
   aves.InitBuffer();

   int depth = fcnMat.Depth();//numHiddenLayer + 1;
   int weightSize = fcnMat.wSize;//wIn*wHidden + wOut*wHidden + (numHiddenLayer - 1)*wHidden*wHidden;
   int biasSize = fcnMat.bSize;// wOut + numHiddenLayer*wHidden;
   int paraSize = weightSize + biasSize;

   float* loc = fcnMat.Mats();//arrangeBuffer(wIn, wOut, wHidden, numHiddenLayer);// new float[paraSize];
   float* biasLoc = loc + weightSize; //这个计算没问题。sizeof float = 64， weightSize = 0x500

   for (int i = 0; i < numHiddenLayer; i++) {
	   network.AddFCNlayer(fcnMat.Width(i+1));
   }


   network.Finish(loc, weightSize, paraSize);
   //network.Finish(fcnMat);
   //network.Finish(loc, biasLoc);
   fcnMat.PassCtrlToShader();
   fcnMat.PassDataToShader();
   //fcnData.PassDataToShader();
//===================================================================================/////////////////
   quadMat.InitBuffer();
   quadMat.InitData();

   depth = quadNoHiddenLayer + 1;
   weightSize = quadIn*quadWHidden + quadOut*quadWHidden + (quadNoHiddenLayer - 1)*quadWHidden*quadWHidden;
   biasSize = quadOut+ quadNoHiddenLayer*quadWHidden;
   paraSize = weightSize + biasSize;

   float* quadWloc = quadMat.Mats();
   float* quadBLoc = quadWloc + weightSize; 
   for (int i = 0; i < quadNoHiddenLayer; i++) {
	   quadNetwork.AddFCNlayer(quadWHidden);
   }
   //quadNetwork.Finish(quadWloc, quadBLoc);
   //quadNetwork.Finish(quadMat);
//=========================================================================
 //fcnMat.ReadFromFile("mats/debugMED/mats201910141008.txt"/*RGB BRDF, 12*12 */);
 //fcnMat.ReadFromFile("mats/debugMED/mats201910210536.txt"/*RGB BRDF, 12*16 */);
 // fcnMat.ReadFromFile("mats/debugMED/mats201910262046.txt"/*RGB BRDF, 12*12, better one */); 


 //"mats/debugMED/mats201910091717.txt"/*trained from sphere*/
 //"mats/debugMED/mats201910120946.txt"/*RGB Phong, 6*10 */
 //"mats/debugMED/mats201910121423.txt"/*RGB Phong, 12*10 */
 //"mats/debugMED/mats201910111017.txt"/*RGB Phong, 5*8 */

   //for (int i = 0; i < 10; i++) {
	  // mats[i] = float(i) / 9.f;
   //}
   //GLuint matLoc = 3;
   //glGenBuffers(1, &matSSBO);
   //glBindBuffer(GL_SHADER_STORAGE_BUFFER, matSSBO);
   //glBufferData(GL_SHADER_STORAGE_BUFFER, 10*sizeof(float), &mats[0], GL_DYNAMIC_COPY);
   //glBindBufferBase(GL_SHADER_STORAGE_BUFFER, matLoc, matSSBO);
   //glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
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
	  case 'i':
	  case 'I':
		  fcnMat.InitData();
		  quadMat.InitData();
		  break;

	  case 'b':
	  case 'B':
		  testQuad = FALSE;
		  loadBuffer();
		  break;

	  case 's':
	  case 'S':
		  testQuad = FALSE;
		  splitData(num_sample);
		  break;

	  case 'w':
	  case 'W':
		  trainFlag = false;
		  fcnMat.WriteToFile();
		  break;

	 // case '0': fcnData.UpdateBatch(0); break;
	 // case '1': fcnData.UpdateBatch(1);break;
	 // case '2': fcnData.UpdateBatch(2);break;
	 // case '3': fcnData.UpdateBatch(3);break;
	 // case '4': fcnData.UpdateBatch(4);break;
	 // case '5': fcnData.UpdateBatch(5);break;
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

//only for debug
void mouse(int button, int state, int x, int y)
{
   ImGui_ImplGlut_MouseButtonCallback(button, state);

   //if (button == 0) {
	  // int mouseX = x;
	  // int mouseY = y;
	  // const int buffer_width = glutGet(GLUT_WINDOW_WIDTH);
	  // const int buffer_height = glutGet(GLUT_WINDOW_HEIGHT);

	  // glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	  // glReadBuffer(GL_COLOR_ATTACHMENT0);
	  // glPixelStorei(GL_PACK_ALIGNMENT, 0);
	  // glReadPixels(mouseX, buffer_height - mouseY, 1, 1, GL_RGBA, GL_FLOAT, bufferValue[0]);

	  // glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	  // glReadBuffer(GL_COLOR_ATTACHMENT1);
	  // glPixelStorei(GL_PACK_ALIGNMENT, 1);
	  // glReadPixels(mouseX, buffer_height - mouseY, 1, 1, GL_RGBA, GL_FLOAT, bufferValue[1]);

	  // glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	  // glReadBuffer(GL_COLOR_ATTACHMENT2);
	  // glPixelStorei(GL_PACK_ALIGNMENT, 2);
	  // glReadPixels(mouseX, buffer_height - mouseY, 1, 1, GL_RGBA, GL_FLOAT, bufferValue[2]);

	  // glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	  // glReadBuffer(GL_COLOR_ATTACHMENT3);
	  // glPixelStorei(GL_PACK_ALIGNMENT, 3);
	  // glReadPixels(mouseX, buffer_height - mouseY, 1, 1, GL_RGBA, GL_FLOAT, bufferValue[3]);

	  // glBindFramebuffer(GL_FRAMEBUFFER, 0);

	  // std::cout << std::setprecision(4) << "x: " << x << ", y: " << y << std::endl;
	  // std::cout << std::setprecision(4) << "buffer0:\t" << bufferValue[0][0] << "\t" << bufferValue[0][1] << "\t" << bufferValue[0][2] << "\t" << bufferValue[0][3] << std::endl;
	  // std::cout << std::setprecision(4) << "buffer1:\t" << bufferValue[1][0] << "\t" << bufferValue[1][1] << "\t" << bufferValue[1][2] << "\t" << bufferValue[1][3] << std::endl;
	  // std::cout << std::setprecision(4) << "buffer2:\t" << bufferValue[2][0] << "\t" << bufferValue[2][1] << "\t" << bufferValue[2][2] << "\t" << bufferValue[2][3] << std::endl;
	  // std::cout << std::setprecision(4) << "buffer3:\t" << bufferValue[3][0] << "\t" << bufferValue[3][1] << "\t" << bufferValue[3][2] << "\t" << bufferValue[3][3] << std::endl;
   //}


   //else {
	  // int mouseX = x;
	  // int mouseY = y;
	  // const int buffer_width = glutGet(GLUT_WINDOW_WIDTH);
	  // const int buffer_height = glutGet(GLUT_WINDOW_HEIGHT);
	  // int index = mouseX * buffer_height + (buffer_height - mouseY);
	  // std::cout << "X: " << std::endl;
	  // std::cout << std::setprecision(4) << "normal:\t" << bufferX[index][0] << "\t" << bufferX[index][1] << "\t" << bufferX[index][2] << std::endl;
	  // std::cout << std::setprecision(4) << "view:\t" << bufferX[index][3] << "\t" << bufferX[index][4] << "\t" << bufferX[index][5] << std::endl;
	  // std::cout << std::setprecision(4) << "light:\t" << bufferX[index][6] << "\t" << bufferX[index][7] << "\t" << bufferX[index][8] << std::endl;
	  // std::cout << std::setprecision(4) << "texCoord:\t" << bufferX[index][9] << "\t" << bufferX[index][10] << std::endl;
	  // std::cout << "Y: " << std::endl;
	  // std::cout << std::setprecision(4) << "RGBA:\t" << bufferY[index][0] << "\t" << bufferY[index][1] << "\t" << bufferY[index][2] << "\t" << bufferY[index][3] << std::endl;
	  // std::cout << std::endl;
   //}
}

int main (int argc, char **argv)
{
   glutInit(&argc, argv); 
   glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
   glutInitWindowPosition (5, 5);
   glutInitWindowSize(700,700);// (1280, 720);
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


