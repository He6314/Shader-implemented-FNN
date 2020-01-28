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

//记得清理ControlPars：现在的这个不需要了，要么再整合一些别的数据，要么直接删掉
//记得改：直接返回变量不安全；

const int MAX_WIDTH = 30;
const int MAX_DEPTH = 30;

struct ControlParas {
	int depth = MAX_WIDTH;
	int weightSize = MAX_WIDTH * MAX_WIDTH * MAX_DEPTH;
};

class FcnSSBO {
public:
	int wIn;
	int wOut;
	int numHiddenLayer;
	int wHidden;

	int wSize;
	int bSize;
	int matSize;

	float *inputAve, *outputAve;

	int Depth() { return paras.depth; }

	int Width(int nbLayer) { return layerWidths[nbLayer]; }

	int Index(bool isBias, int nbLayer, int nb1, int nb2=0) {
		if (!isBias) {
			int wShift = 0;
			for (int n = 1; n < nbLayer+1; n++) {
				wShift += layerWidths[n - 1] * layerWidths[n];
			}
			int wTop = layerWidths[nbLayer+1];
			int wBottom = layerWidths[nbLayer];
			return wShift + nb1*wTop + nb2;
		}
		else {
			int bShift = 0;
			for (int n = 1; n < nbLayer + 1; n++) {
				bShift += layerWidths[n];
			}
			return paras.weightSize + bShift + nb1;
		}
	}

	void InitData() {
		static std::random_device rd;
		static std::default_random_engine generator(rd());
		float InitWeightParams[2] = { -0.95f, 0.95f };
		std::normal_distribution<float> n_dist(InitWeightParams[0], InitWeightParams[1]);
		std::uniform_real_distribution<float> u_dist(InitWeightParams[0], InitWeightParams[1]);

		for (int i = 0; i < matSize; i++) {
			mats[i] = u_dist(generator);//float(i);//
		}
	}


	float* Mats() {
		float* pt = mats;
		return pt;
	}

	float* Mats(int loc) {
		float* pt = mats + loc;
		return pt;
	}

	void PassCtrlToShader() {
		// sizeof(ControlParas)
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, paraUBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ControlParas), &paras.depth, GL_DYNAMIC_COPY);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, paraLoc, paraUBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, widthSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, (paras.depth + 1) * sizeof(int), &layerWidths[0], GL_DYNAMIC_COPY);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, widthLoc, widthSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void PassDataToShader() {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, matSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, matSize * sizeof(float), &mats[0], GL_DYNAMIC_COPY);
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

	void InitBuffer(int i = 0) {
		paraLoc = 2 + i * 10;
		glGenBuffers(1, &paraUBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, paraUBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ControlParas), &paras.depth, GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, paraLoc, paraUBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		widthLoc = 3 + i * 10;
		glGenBuffers(1, &widthSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, widthSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, (paras.depth + 1) * sizeof(int), &layerWidths[0], GL_DYNAMIC_COPY);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, widthLoc, widthSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		matLoc = 4 + i * 10;
		glGenBuffers(1, &matSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, matSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, matSize * sizeof(float), &mats[0], GL_DYNAMIC_COPY);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, matLoc, matSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void WriteToFile() {
		ofstream file;

		time_t t = std::time(nullptr);
		tm local = *std::localtime(&t);
		std::ostringstream filename;
		filename << "mats/debugMED/mats" << std::put_time(&local, "%Y%m%d%H%M") << ".txt";
		file.open(filename.str());

		file << "DEPTH " << paras.depth << "\n";
		file << "BIASLOC " << paras.weightSize << "\n";
		file << "WIDTH ";
		for (int i = 0; i < paras.depth + 1; i++)
			file << layerWidths[i] << " ";
		file << "\n\n";

		file << "INPUT_AVE ";
		for (int i = 0; i < layerWidths[0]; i++)
			file << inputAve[i] << " ";
		file << "\n\n";

		file << "OUTPUT_AVE ";
		for (int i = 0; i < layerWidths[paras.depth]; i++)
			file << outputAve[i] << " ";
		file << "\n\n";

		file << "DATA " << matSize << "\n";
		for (int i = 0; i < matSize; i++)
			file << mats[i] << " ";
		file << "\n";

		std::cout << "Written in file: " << filename.str() << std::endl;
		file.close();
	}

	void ReadFromFile(string filename) {
		ifstream file(filename);
		if (!file.is_open())
			printf("File doesn't exist！\n");

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
							layerWidths[i] = atof(numLine.substr(0, numLine.find(' ')).c_str());
							numLine = numLine.substr(numLine.find(' ') + 1);
						}
						wIn = layerWidths[0];
						wOut = layerWidths[paras.depth];
						wHidden = layerWidths[1];
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
					mats = new float[matSize]();
					for (int i = 0; i < matSize; i++) {
						mats[i] = atof(matLine.substr(0, matLine.find(' ')).c_str());
						matLine = matLine.substr(matLine.find(' ') + 1);
					}
					delete[] matBuffer;
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
		numHiddenLayer = MAX_DEPTH - 1;

		Malloc();
	}
	FcnSSBO(int dim_in, int dim_out, int dim_hidden, int num_hidden) {
		wIn = dim_in;
		wOut = dim_out;
		wHidden = dim_hidden;
		numHiddenLayer = num_hidden;

		Malloc();
	}
	~FcnSSBO() {
		if (mats != 0)delete[] mats;
		if (layerWidths != 0) delete[] layerWidths;
	}
private:
	ControlParas paras;
	int* layerWidths;

	float* mats;
	//int matSize;

	GLint paraLoc, widthLoc, matLoc;
	GLuint paraUBO = -1, widthSSBO = -1, matSSBO = -1;

	void Malloc() {
		paras.depth = numHiddenLayer + 1;
		layerWidths = new int[numHiddenLayer + 2]();
		layerWidths[0] = wIn;
		layerWidths[paras.depth] = wOut;
		for (int i = 1; i < paras.depth; i++) {
			layerWidths[i] = wHidden;
			//layerWidths[i] = wHidden + i;//！！
		}

		wSize = 0;
		bSize = 0;
		for (int i = 0; i < paras.depth; i++) {
			wSize += layerWidths[i] * layerWidths[i + 1];
			bSize += layerWidths[i + 1];
		}
		matSize = wSize + bSize;
		paras.weightSize = wSize;

		mats = new float[matSize]();

		inputAve = new float[wIn]();
		outputAve = new float[wOut]();
	}
};


struct AveVectorUBO
{
	int inDim, outDim;
	float* aveIn;
	float* aveOut;
	GLuint inUbo, outUbo;
	GLuint inLoc = -1, outLoc = -1;

	void PassDataToShader() {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, inUbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, inDim * sizeof(float), &aveIn[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, inLoc, inUbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, outUbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, outDim * sizeof(float), &aveOut[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, outLoc, outUbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}
	void PassDataToShader(float* inVec, float* outVec) {
		for (int i = 0; i < inDim; i++) {
			aveIn[i] = inVec[i];
		}
		for (int i = 0; i < outDim; i++) {
			aveOut[i] = outVec[i];
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, inUbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, inDim * sizeof(float), &aveIn[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, inLoc, inUbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, outUbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, outDim * sizeof(float), &aveOut[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, outLoc, outUbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	AveVectorUBO() {};
	AveVectorUBO(float* inVec, int inDim, float* outVec, int outDim) : inDim(inDim), outDim(outDim) {
		aveIn = new float[inDim]();
		aveOut = new float[outDim]();
	}

	void InitBuffer() {
		inLoc = 5;
		outLoc = 6;
		glGenBuffers(1, &inUbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, inUbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, inDim, &aveIn[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, inLoc, inUbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		glGenBuffers(1, &outUbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, outUbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, outDim, &aveOut[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, outLoc, outUbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	~AveVectorUBO() {
		if (aveIn != 0)
			delete[] aveIn;
		if (aveOut != 0)
			delete[] aveOut;
	}
};
