#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

struct TransMatrices
{
	glm::mat4 P;
	glm::mat4 V;
	glm::mat4 M;
	glm::mat4 PV;
	glm::mat4 PVM;
	glm::mat4 Vinv;
	glm::vec4 World_CamPos;
	glm::vec2 Viewport;
};

struct TransUBO
{
	TransMatrices mats;
	unsigned int ubo;
	GLuint binding_loc;

	void InitBuffer();
	void PassDataToShader();
	TransUBO();
};