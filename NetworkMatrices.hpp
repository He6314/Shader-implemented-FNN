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
#include <vector> //here

using namespace std;
//����ControlPars�����ڵ��������Ҫ�ˣ�Ҫô������һЩ������ݣ�Ҫôֱ��ɾ��
//����AveVector�����Բ���SSBO����
//�ǵøģ�ֱ�ӷ��ر�������ȫ��
//������AveVector������SSBO��

const int MAX_WIDTH = 30;
const int MAX_DEPTH = 30;

struct ControlParas {
	int depth = MAX_WIDTH;
	int weightSize = MAX_WIDTH * MAX_WIDTH * MAX_DEPTH;
	int paraSize = MAX_WIDTH * MAX_WIDTH * MAX_DEPTH + MAX_WIDTH * MAX_DEPTH;
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

		for (int i = matSize; i < 4 * matSize; i++) {
			mats[i] = 0.0;// float(i) / matSize; //0.0;//0.1 * 
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

	float* Weights() {
		float* pt = mats;
		return pt;
	}

	float* Bias() {
		float* pt = mats;
		return pt + wSize;
	}

	void printMat(int loc) {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, matSSBO);
		GLfloat *ptr;
		ptr = (GLfloat *)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE); //GL_READ_WRITE?

		int dataLoc = 0 * paras.paraSize;
		int weightShift = 0;
		int biasShift = paras.weightSize;
		for (int i = 0; i < paras.depth; i++) {
			int wTop = layerWidths[i];
			int wBottom = layerWidths[i+1];
			for (int j = 0; j < wBottom; j++) {
				for (int k = 0; k < wTop; k++)
				{
					cerr << fixed << setprecision(5) << ptr[dataLoc + weightShift + j * wTop + k] << "  ";
				}
				cerr << "|";
				cerr << fixed << setprecision(5) << ptr[dataLoc + biasShift + j] << endl;
			}
			cerr << "---------------------------------" << endl;
			weightShift += wTop * wBottom;
			biasShift += wBottom;
		}//0


		cerr << "=============================================================" << endl;
		dataLoc = 1 * paras.paraSize;
		weightShift = 0;
		biasShift = paras.weightSize;
		for (int i = 0; i < paras.depth; i++) {
			int wTop = layerWidths[i];
			int wBottom = layerWidths[i + 1];
			for (int j = 0; j < wBottom; j++) {
				for (int k = 0; k < wTop; k++)
				{
					cerr << fixed << setprecision(5) << ptr[dataLoc + weightShift + j * wTop + k] << "  ";
				}
				cerr << "|";
				cerr << fixed << setprecision(5) << ptr[dataLoc + biasShift + j] << endl;
			}
			cerr << "---------------------------------" << endl;
			weightShift += wTop * wBottom;
			biasShift += wBottom;
		}//1

		cerr << "=============================================================" << endl;
		dataLoc = 2 * paras.paraSize;
		weightShift = 0;
		biasShift = paras.weightSize;
		for (int i = 0; i < paras.depth; i++) {
			int wTop = layerWidths[i];
			int wBottom = layerWidths[i + 1];
			for (int j = 0; j < wBottom; j++) {
				for (int k = 0; k < wTop; k++)
				{
					cerr << fixed << setprecision(5) << ptr[dataLoc + weightShift + j * wTop + k] << "  ";
				}
				cerr << "|";
				cerr << fixed << setprecision(5) << ptr[dataLoc + biasShift + j] << endl;
			}
			cerr << "---------------------------------" << endl;
			weightShift += wTop * wBottom;
			biasShift += wBottom;
		}//2

		cerr << "=============================================================" << endl;
		dataLoc = 3 * paras.paraSize;
		weightShift = 0;
		biasShift = paras.weightSize;
		for (int i = 0; i < paras.depth; i++) {
			int wTop = layerWidths[i];
			int wBottom = layerWidths[i + 1];
			for (int j = 0; j < wBottom; j++) {
				for (int k = 0; k < wTop; k++)
				{
					cerr << fixed << setprecision(5) << ptr[dataLoc + weightShift + j * wTop + k] << "  ";
				}
				cerr << "|";
				cerr << fixed << setprecision(5) << ptr[dataLoc + biasShift + j] << endl;
			}
			cerr << "---------------------------------" << endl;
			weightShift += wTop * wBottom;
			biasShift += wBottom;
		}//3
		cerr << "###################################" << endl;

		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
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
		glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * matSize * sizeof(float), &mats[0], GL_DYNAMIC_COPY);
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
		glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * matSize * sizeof(float), &mats[0], GL_DYNAMIC_COPY);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, matLoc, matSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		//here
		//InitData();
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
			printf("File doesn't exist��\n");

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
					mats = new float[4 * matSize]();
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
			//layerWidths[i] = wHidden + i;//debug
		}

		wSize = 0;
		bSize = 0;
		for (int i = 0; i < paras.depth; i++) {
			wSize += layerWidths[i] * layerWidths[i + 1];
			bSize += layerWidths[i + 1];
		}
		matSize = wSize + bSize;
		paras.weightSize = wSize;
		paras.paraSize = matSize;

		mats = new float[4 * matSize](); 
		//arrangement: (weights->biases) -> (dWeight->dBias) -> moments->velocity

		inputAve = new float[wIn]();
		outputAve = new float[wOut]();
	}
};

//Can be optimized: AveVectorUBO can be part// of dataSSBO
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

	//here
	void AssignData(float* inVec, float* outVec) {
		for (int i = 0; i < inDim; i++) {
			aveIn[i] = inVec[i];
		}
		for (int i = 0; i < outDim; i++) {
			aveOut[i] = outVec[i];
		}
	};
	//here
	void Reshape(int InDim, int OutDim) {
		inDim = InDim;
		outDim = OutDim;
		
		if (NULL != aveIn) {
			delete[] aveIn;
			delete[] aveOut;
		}
		aveIn = new float[inDim]();
		aveOut = new float[outDim]();
		InitBuffer();
	}

	void InitBuffer(int i = 0) {
		inLoc = 5 + i * 10;
		outLoc = 6 + i * 10;
		
		//here
		if(inUbo<=0)
			glGenBuffers(1, &inUbo);
		if(outUbo<=0)
			glGenBuffers(1, &outUbo);
		
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, inUbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, inDim, &aveIn[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, inLoc, inUbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, outUbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, outDim, &aveOut[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, outLoc, outUbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}
	
	AveVectorUBO() {};
	AveVectorUBO(int inDim, int outDim) : inDim(inDim), outDim(outDim) {
		aveIn = new float[inDim]();
		aveOut = new float[outDim]();
	}
	//here
	AveVectorUBO(float* inVec, int inDim, float* outVec, int outDim) : inDim(inDim), outDim(outDim) {
		aveIn = new float[inDim]();
		aveOut = new float[outDim]();
	}

	~AveVectorUBO() {
		//here
		if(NULL != aveIn)
			delete[] aveIn;
		if (NULL != aveOut)
			delete[] aveOut;
	}
};

class DataSSBO {
public:

	float* dataID; //int
	GLuint trainDataSSBO, validDataSSBO, idSSBO;

	DataSSBO(int inputDim, int outputDim, float trainRatio = 0.8, int indexDim = 3) :
		inputDim(inputDim), outputDim(outputDim), trainRatio(trainRatio), indexDim(indexDim)
	{
		dataDim = inputDim + outputDim + indexDim;
	};

	//**Temporarily Abandoned: Not sure what else should be done with numSample.
	//DataSSBO(int inputDim, int outputDim, int numSample, float trainRation) :
	//	inputDim(inputDim), outputDim(outputDim), numSample(numSample), trainRatio(trainRatio)
	//{
	//	dataDim = inputDim + outputDim + indexDim;
	//	numTrain = numSample * trainRatio;
	//	numValid = numSample - numTrain;
	//	Malloc();
	//};

	~DataSSBO() {
		delete[] trainData;
		delete[] validData;
		delete[] data;
		delete[] dataID;
	}

	void InitBuffer(int i = 0) {
		if (NULL == data || numSample <= 0) {
			cerr << "Data error: Memory is not located or Samples are not assigned." << endl;
		}
		else {
			trainDataLoc = 7 + i * 10;
			glGenBuffers(1, &trainDataSSBO);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, trainDataSSBO);
			glBufferData(GL_SHADER_STORAGE_BUFFER, (numTrain*dataDim) * sizeof(int), trainData, GL_DYNAMIC_COPY);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, trainDataLoc, trainDataSSBO);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

			validDataLoc = 8 + i * 10;
			glGenBuffers(1, &validDataSSBO);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, validDataSSBO);
			glBufferData(GL_SHADER_STORAGE_BUFFER, (numValid*dataDim) * sizeof(float), validData, GL_DYNAMIC_COPY);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, validDataLoc, validDataSSBO);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

			AveVecs.InitBuffer(i);

			idLoc = 9 + i * 10;
			glGenBuffers(1, &idSSBO);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, idSSBO);
			glBufferData(GL_SHADER_STORAGE_BUFFER, (batch_size+1) * sizeof(float), dataID, GL_DYNAMIC_COPY);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, idLoc, idSSBO);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		}
	};

	void EnterData(float* Inputs, float* Outputs, int iX, int iY) {
		int i = 0;
		float* tempData = new float[dataDim]();

		for (i = 0; i < inputDim; i++) {
			tempData[i] = Inputs[i];
		}
		for (i = 0; i < outputDim; i++) {
			tempData[inputDim + i] = Outputs[i];
		}
		tempData[inputDim + outputDim] = iX;
		tempData[inputDim + outputDim + 1] = iY;

		tempData[dataDim - 1] = numSample;

		unsortedData.push_back(tempData);
		numSample++;
	};

	void InitData() {
		for (int i = 0; i < unsortedData.size(); i++) {
			delete[] unsortedData[i];
		}
		vector<float*>().swap(unsortedData);

		delete[] trainData;
		delete[] validData;
		delete[] data;

		numTrain = 0;
		numValid = 0;
		numSample = 0;
	};

	void SplitData(bool shuffle = true) {
		numTrain = numSample * trainRatio;
		numValid = numSample - numTrain;
		Malloc();

		std::vector<int> index;
		for (int i = 0; i < numSample; i++) index.push_back(i);
		if(shuffle)
			std::random_shuffle(index.begin(), index.end());
		
		float* aveVec = new float[dataDim]();

		int curIdx = 0;
		for (int i = 0; i < numTrain; i++) {
			curIdx = index[i];
			for (int j = 0; j < dataDim; j++) {
				data[i*dataDim + j] = unsortedData[curIdx][j];
				aveVec[j] += unsortedData[curIdx][j];
			}
		}
		cerr << "Train data ready: " << numTrain << std::endl;

		for (int j = 0; j < inputDim; j++) {
			aveVec[j] /= numTrain;
			AveVecs.aveIn[j] = aveVec[j];
		}
		for (int j = 0; j < outputDim; j++) {
			aveVec[j + inputDim] /= numTrain;
			AveVecs.aveOut[j] = aveVec[j + inputDim];
		}
		cerr << "Average Vectors Ready." << endl;

		aveVec[inputDim + outputDim] /= numTrain;
		aveVec[inputDim + outputDim + 1] /= numTrain;
		aveVec[inputDim + outputDim + 2] /= numTrain;
		cerr << "Debug: Ave Coords:\t" << aveVec[inputDim + outputDim] << "\t" << aveVec[inputDim + outputDim + 1] << endl;
		cerr << "Ave Index:" << aveVec[inputDim + outputDim + 2] << endl;

		//float maxTest;
		for (int i = 0; i < numValid; i++) {
			curIdx = index[i + numTrain];
			for (int j = 0; j < dataDim; j++) {
				data[(i + numTrain)*dataDim + j] = unsortedData[curIdx][j];
			}
		}
		cerr << "Validation data ready: " << numValid << endl;

		InitBuffer();

		/*for (int i = 0; i < 1000; i++) {
			cerr << data[(i + 1)*dataDim - 1] << endl;
		}*/
	};

	//Shouldn't use GL_DYNAMIC_COPY: shader should not have the right to modify data
	void PassDataToShader() {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, trainDataSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, (numTrain * dataDim) * sizeof(float), &trainData[0], GL_DYNAMIC_COPY);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, trainDataLoc, trainDataSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, validDataSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, (numValid * dataDim) * sizeof(float), &validData[0], GL_DYNAMIC_COPY);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, validDataLoc, validDataSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		AveVecs.PassDataToShader();
	}

	void BatchSize(int batchSize) {
		batch_size = batchSize;
	}

	int BatchSize() { return batch_size; }

	//May be smaller: index may not be used.
	void UpdateBatch(int batch_num) {
		dataID[0] = batch_num * batch_size;
		for (int i = 0; i < batch_size+1; i++) {
			dataID[i + 1] = 0;// data[(batch_num*batch_size + i + 1)*dataDim - 1];
		}
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, idSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, (batch_size+1) * sizeof(float), &dataID[0], GL_DYNAMIC_COPY);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, idLoc, idSSBO);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

private:
	int inputDim;
	int outputDim;
	int dataDim;
	int indexDim = 3;
	//Data will be arranged as input-output-index(x,y,no.in_sequence)

	int numTrain;
	int numValid;
	int numSample;
	float trainRatio;

	int batch_size = 8;
	int curBatch = 0;

	vector<float*> unsortedData;
	float* data;
	float* trainData;
	float* validData;
	//dataid
	
	GLint trainDataLoc, validDataLoc, idLoc;
	//ssbo

	AveVectorUBO  AveVecs;

	void Malloc() {
		if (numSample <= 0) { 
			cerr << "Data error: No sample found." << endl;
		}
		else {
			if (NULL != data) {
				//delete[] trainData;
				//delete[] validData;
				delete[] data;
			}
			data = new float[dataDim* numSample]();
			trainData = data;
			validData = data + (dataDim*numTrain);
			
			dataID = new float[batch_size + 1];//int

			AveVecs.Reshape(inputDim, outputDim);
		}
	}
};