#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include "TransMatrices.h"

TransUBO::TransUBO()
{
	binding_loc = 1;
}

void TransUBO::InitBuffer() {
	glGenBuffers(1, &ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(TransMatrices), &mats.P[0], GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_UNIFORM_BUFFER, binding_loc, ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void TransUBO::PassDataToShader()
{
	mats.PV = mats.P*mats.V;
	mats.PVM = mats.PV*mats.M;
	mats.Vinv = glm::inverse(mats.V);
	mats.World_CamPos = mats.Vinv[3];

	glBindBuffer(GL_UNIFORM_BUFFER, ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(TransMatrices), &mats.P[0]);
	glBindBufferBase(GL_UNIFORM_BUFFER, binding_loc, ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}