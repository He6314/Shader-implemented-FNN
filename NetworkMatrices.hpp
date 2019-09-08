#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <fstream>
#include <sstream>
#include <ctime>

const int MAX_WIDTH = 30;
const int MAX_DEPTH = 30;

struct ControlParas {
	int depth = MAX_WIDTH;
	int weightSize = MAX_WIDTH * MAX_WIDTH * MAX_DEPTH;
	int width[MAX_DEPTH];
};

class FcnSSBO {
public:
	int wIn;
	int wOut;
	int numHiddenLayer;
	int wHidden;
	float* Mats() {
		float* pt = mats;
		return pt;
	}

	void PassCtrlToShader() {
		// sizeof(ControlParas)

		for (int i = 0; i < MAX_DEPTH; i++) {
			paras.width[i] = i + 1;
		}

		glBindBuffer(GL_UNIFORM_BUFFER, paraUBO);
		glBufferData(GL_UNIFORM_BUFFER, 128, &paras.depth, GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_UNIFORM_BUFFER, paraLoc, paraUBO);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	void PassDataToShader()	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, matSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, matSize*sizeof(float), &mats[0], GL_DYNAMIC_COPY);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, matLoc, matSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void Reshape(int dim_in, int dim_out, int dim_hidden, int num_hidden) {
		if (mats != 0) {
			delete[] mats;
		}
		wIn = dim_in;
		wOut = dim_out;
		wHidden = dim_hidden;
		numHiddenLayer = num_hidden;

		Malloc();
		InitBuffer();
		PassCtrlToShader();
	}
	
	void InitBuffer() {
		paraLoc = 2;
		glGenBuffers(1, &paraUBO);
		glBindBuffer(GL_UNIFORM_BUFFER, paraUBO);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(ControlParas), &paras.depth, GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_UNIFORM_BUFFER, paraLoc, paraUBO);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		matLoc = 3;
		glGenBuffers(1, &matSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, matSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, matSize, &mats[0], GL_DYNAMIC_COPY);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, matLoc, matSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void WriteToFile() {
		ofstream file;

		time_t t = std::time(nullptr);
		tm local = *std::localtime(&t);
		std::ostringstream filename;
		filename << "mats/mats" << std::put_time(&local, "%Y%m%d%H%M") << ".txt";
		file.open(filename.str());

		file << "DEPTH " << paras.depth << "\n";
		file << "BIASLOC " << paras.weightSize << "\n";
		file << "WIDTH ";
		for (int i = 0; i < paras.depth + 1; i++)
			file << paras.width[i] << " ";
		file << "\n\n";

		file << "DATA "<< matSize <<"\n";
		for (int i = 0; i < matSize; i++)
			file << mats[i] << " ";
		file << "\n";

		std::cout << "Written in file: " << filename.str() << std::endl;
		file.close();
	}

	void ReadFromFile(string filename) {
		ifstream file(filename);
		if (!file.is_open())
			printf("File doesn't exist£¡\n");

		else {
			char charBuffer[100];
			while (!file.eof()) {
				file.getline(charBuffer, 100, '\n');
				string strLine = string(charBuffer);
				if (strLine.find("DATA") == -1) {
					if (strLine.find("DEPTH") != -1) {
						paras.depth = atof(strLine.substr(strLine.find(' ') + 1).c_str());
						numHiddenLayer = paras.depth - 1;
					}
					if (strLine.find("BIASLOC") != -1) {
						paras.weightSize = atof(strLine.substr(strLine.find(' ') + 1).c_str());
					}
					if (strLine.find("WIDTH") != -1) {
						string numLine = strLine.substr(strLine.find(' ') + 1);
						for (int i = 0; i < paras.depth + 1; i++) {
							paras.width[i] = atof(numLine.substr(0, numLine.find(' ')).c_str());
							numLine = numLine.substr(numLine.find(' ') + 1);
						}
						wIn = paras.width[0];
						wOut = paras.width[paras.depth];
						wHidden = paras.width[1];
					}
				}
				else {
					matSize = atof(strLine.substr(strLine.find(' ') + 1).c_str());
					char* matBuffer = new char[matSize * 11];

					file.getline(matBuffer, matSize * 11, '\n');
					string matLine = string(matBuffer);
					if (mats != 0) {
						delete[] mats;
					}
					mats = new float[matSize];
					for (int i = 0; i < matSize; i++) {
						mats[i] = atof(matLine.substr(0, matLine.find(' ')).c_str());
						matLine = matLine.substr(matLine.find(' ') + 1);
					}
					delete matBuffer;
					break;
				}
			}

			file.close();

			InitBuffer();
			PassCtrlToShader();
			PassDataToShader();

			std::cout << "Reading completed." << std::endl;
		}
	}

	FcnSSBO() {
		wIn = MAX_WIDTH;
		wOut = MAX_WIDTH;
		wHidden = MAX_WIDTH;
		numHiddenLayer = MAX_DEPTH-1;

		Malloc();
		//InitBuffer();
	}
	FcnSSBO(int dim_in, int dim_out, int dim_hidden, int num_hidden) {
		wIn = dim_in;
		wOut = dim_out;
		wHidden = dim_hidden;
		numHiddenLayer = num_hidden;

		Malloc();
		//InitBuffer();
	}
	~FcnSSBO() {
		delete[] mats;
	}
private:
	ControlParas paras;
	float* mats;
	int matSize;

	GLint paraLoc;
	GLint matLoc;
	unsigned int paraUBO = -1;
	unsigned int matSSBO = -1;

	void Malloc() {
		int wSize = wIn*wHidden + wOut*wHidden + (numHiddenLayer-1)*wHidden*wHidden;
		int bSize = wOut + numHiddenLayer*wHidden;
		matSize = wSize + bSize;

		paras.depth = numHiddenLayer + 1;
		paras.width[0] = wIn;
		paras.width[paras.depth] = wOut;
		for (int i = 1; i < numHiddenLayer+1; i++) {
			paras.width[i] = wHidden;
		}
		paras.weightSize = wSize;

		mats = new float[matSize]();
	}
};

struct AveVector {
	float aveIn[MAX_WIDTH];
	float aveOut[MAX_WIDTH];
};
struct AveVectorUBO
{
	AveVector aves;
	unsigned int ubo;
	GLuint binding_loc;

	void PassDataToShader() {
		glBindBuffer(GL_UNIFORM_BUFFER, ubo);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(AveVector), &aves.aveIn[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_UNIFORM_BUFFER, binding_loc, ubo);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}
	void PassDataToShader(float* inVec, int inDim, float* outVec, int outDim) {
		for (int i = 0; i < inDim; i++) {
			aves.aveIn[i] = inVec[i];
		}
		for (int i = 0; i < outDim; i++) {
			aves.aveOut[i] = outVec[i];
		}
		glBindBuffer(GL_UNIFORM_BUFFER, ubo);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(AveVector), &aves.aveIn[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_UNIFORM_BUFFER, binding_loc, ubo);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	AveVectorUBO() {
		binding_loc = 4;
		glGenBuffers(1, &ubo);
		glBindBuffer(GL_UNIFORM_BUFFER, ubo);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(AveVector), &aves.aveIn[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_UNIFORM_BUFFER, binding_loc, ubo);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}
	AveVectorUBO(float* inVec, int inDim, float* outVec, int outDim) {
		for (int i = 0; i < inDim; i++) {
			aves.aveIn[i] = inVec[i];
		}
		for (int i = 0; i < outDim; i++) {
			aves.aveOut[i] = outVec[i];
		}
		
		binding_loc = 4;
		glGenBuffers(1, &ubo);
		glBindBuffer(GL_UNIFORM_BUFFER, ubo);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(AveVector), &aves.aveIn[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_UNIFORM_BUFFER, binding_loc, ubo);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}
};
