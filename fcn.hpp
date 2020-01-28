#pragma once

#include <windows.h>
#include <iostream>
#include <string>
#include <random>
#include <math.h>
#include <vector>

using namespace std;

struct DataNode {
	//float zVec[50];
	//float aVec[50];
	//float dVec[50];

	//float groundTruth[50];

	float* zVec;
	float* aVec;
	float* dVec;

	float* groundTruth;

	int width;
	string name;

	void printValue();
	void printDiff();

	DataNode::DataNode();
	DataNode(string newName, int width);

	void FreeMem();
	~DataNode();
};

struct FcnLayer {
	int activateType = 2;
	//Don't allocate memory here, do it in the network for continuous and stable memory allocation.
	float** weight;
	float* bias;

	float** dMat;
	float* dbVec;
	float** mMat;
	float** vMat;
	float* mbVec;
	float* vbVec;

	//float weight[100][100];
	//float bias[100];
	//float dMat[100][100];
	//float dbVec[100];
	//float mMat[100][100];
	//float vMat[100][100];
	//float mbVec[100];
	//float vbVec[100];
	//==================================

	DataNode* top;
	DataNode* bottom;
	int wTop;
	int wBottom;

	void Forward();
	void Backward();
	void AccuGrad();

	void AdamFixed(int t, float lr);
	void Adam(int t, float beta1, float beta2, float alpha);

	void ClearD();
	void InitData();

	//for debug
	void printPara();
	void printDMat();

	void Malloc(float* weightLoc, float* biasLoc, float* dMatLoc, float* dbVecLoc, float* mMatLoc, float* mbVecLoc, float* vMatLoc, float* vbVecLoc);
	void FreeMem();

	FcnLayer();
	FcnLayer(DataNode* t, DataNode* b);
	~FcnLayer();
};

class Fcn {
private:
	vector<DataNode> nodes;
	vector<FcnLayer> layers;

	int depth;
	int wIn;
	int wOut;
	float** input;
	float** output;
	//float input[100][100];
	//float output[100][100];
	float** testIn;
	float** testOut;

	int dataSize;
	int batchSize;
	int validSize;

	float* dMatLoc;
	float* mMatLoc;
	float* vMatLoc;
	float* dbVecLoc;
	float* mbVecLoc;
	float* vbVecLoc;

	int matSize;
	int noMin;
	float* matPt;
	float* matMin;

	bool finishFlag = FALSE;

public:
	vector<float> trainErrors;
	vector<float> validErrors;
	int maxEpochs = 200;
	const int minEpochs = 10;

	void TrainCPU();
	void TrainCPU_1epoch(int n, float beta1, float beta2, float alpha);
	bool TrainCPU_1batch(int num_batch, int num_epoch, float beta1, float beta2, float alpha);

	float* Evaluate(float* x); //not used

	void ForwardCS();

	void SetTrainData(float** x, float** y, int size, int bSize);
	void SetValidData(float** x, float** y, int size);
	void AddFCNlayer(int width);
	void Finish(float* weightLoc, float* biasLoc);

	int Depth() { return depth; }
	int NumDataNodes() { return depth + 1; }
	int NumHiddenLayer() { return depth - 1; }

	Fcn(int wX, int wY);
	Fcn(float** x, float** y, int wX, int wY, int dataSize, int batchSize);
	~Fcn();
};



///////////////////////////////////////////////////////////////////////
//Implementation
///////////////////////////////////////////////////////////////////////
//float tanh(float x) {
//	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
//}

float activateFunc(float x, int type = 2) {
	switch(type){
	case 0: return (x > 0) ? x : 0; break;//relu
	case 1: return (x > 0) ? x : (0.05*x); break;//leaky relu
	case 2: return tanh(x); break;//tanh 2.0 / (1.0 + exp(-2.0*x)) - 1.0;
	}
};
float dActivateFunc(float x, int type = 2) {
	switch (type) {
		case 0: return float(x>0);//drelu
		case 1: return (x > 0) ? 1.0 : 0.05;//d leaku relu
		case 2: return 1 - tanh(x)*tanh(x);
	}
};

//----------------------------------------
//NODE
//----------------------------------------

void DataNode::FreeMem() {
	delete[] zVec;
	delete[] aVec;
	delete[] dVec;
	delete[] groundTruth;
}

DataNode::DataNode() {
	name = "BAD";
	//cout << "构造node:" << name << endl;
	width = 0;

	zVec = new float[width];
	aVec = new float[width];
	dVec = new float[width];
	groundTruth = new float[width];
}

DataNode::DataNode(string newName, int w) {
	name = newName;
	//cout << "构造node:" << name << endl;
	width = w;

	zVec = new float[width];
	aVec = new float[width];
	dVec = new float[width];
	groundTruth = new float[width];
}

DataNode::~DataNode() {
	//delete[] this->zVec;
	//delete[] this->aVec;
	//delete[] this->dVec;
	//delete[] this->groundTruth;
	//cout << "析构" << name << endl;
}

void DataNode::printValue() {
	cout.width(8);
	cout << endl;
	cout << name << ":\t" << endl;
	for (int i = 0; i < width; i++) {
		cout << zVec[i] << "\t";
	}
	cout << endl;
	for (int i = 0; i < width; i++) {
		cout << aVec[i] << "\t";
	}
	cout << endl;
}
void DataNode::printDiff() {
	cout.width(8);
	cout << endl;
	cout << name << " diff:\t" << endl;
	for (int i = 0; i < width; i++) {
		cout << dVec[i] << "\t";
	}
	cout << endl;
}

//----------------------------------------
//LAYER
//----------------------------------------

void FcnLayer::ClearD() {
	for (int i = 0; i < wBottom; i++) {
		dbVec[i] = 0;
		for (int j = 0; j < wTop; j++) {
			dMat[i][j] = 0;
		}
	}
}

void FcnLayer::InitData() {
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	float InitWeightParams[2] = { -0.95f, 0.95f };
	//std::normal_distribution<float> n_dist(InitWeightParams[0], InitWeightParams[1]);
	std::uniform_real_distribution<float> u_dist(InitWeightParams[0], InitWeightParams[1]);

	for (int i = 0; i < wBottom; i++) {
		bias[i] = u_dist(generator);
		//dbVec[i] = 0;
		//mbVec[i] = 0;
		//vbVec[i] = 0;
		for (int j = 0; j < wTop; j++) {
			weight[i][j] = u_dist(generator);
			//dMat[i][j] = 0;
			//mMat[i][j] = 0;
			//vMat[i][j] = 0;
		}
	}
	//if (wTop == 4)
	//cout << &weight[0][0] << "\t" << &weight[0][0]+sizeof(float) << endl;
	//cout << &weight[0][0] << "\t" << &weight[0][1] << "\t" << &weight[0][2] << "\t" << &weight[0][3] << endl;
	//cout << weight[0][0] << "\t" << weight[0][1] << "\t" << weight[0][2] << "\t" << weight[0][3] << endl;
}

void FcnLayer::Forward() {
	for (int i = 0; i < wBottom; i++) {
		bottom->zVec[i] = 0;
		for (int j = 0; j < wTop; j++) {
			bottom->zVec[i] += weight[i][j] * top->aVec[j];
		}
		bottom->zVec[i] += bias[i];
		bottom->aVec[i] = activateFunc(bottom->zVec[i],this->activateType);
	}
}

void FcnLayer::Backward() {
	for (int j = 0; j < wTop; j++) {
		top->dVec[j] = 0;
		for (int i = 0; i < wBottom; i++) {
			top->dVec[j] += weight[i][j] * bottom->dVec[i];
		}
		top->dVec[j] *= dActivateFunc(top->zVec[j], this->activateType);
	}
}

void FcnLayer::AccuGrad() {
	for (int i = 0; i < wBottom; i++) {
		for (int j = 0; j < wTop; j++) {
			dMat[i][j] += bottom->dVec[i] * top->aVec[j];
		}
		dbVec[i] += bottom->dVec[i];
	}
}

void FcnLayer::AdamFixed(int t, float lr) {
	const float bias_factor = 2.0f;
	float mBeta1 = 0.9f;
	float mBeta2 = 0.9f;//0.999f
	float mAlpha = 0.0002f;//0.0005f
	float mEps = 1e-8;

	float hat1 = 1.0f / (1.0f - pow(mBeta1, t));
	float hat2 = 1.0f / (1.0f - pow(mBeta2, t));

	for (int i = 0; i < wBottom; i++) {
		mbVec[i] = mBeta1* mbVec[i] + (1.0f - mBeta1)*dbVec[i];
		vbVec[i] = mBeta2* vbVec[i] + (1.0f - mBeta2)* dbVec[i] * dbVec[i];
		float deltaB = (hat1* mbVec[i]) / (sqrt(hat2* vbVec[i]) + mEps);
		bias[i] -= bias_factor*mAlpha* deltaB;
		for (int j = 0; j < wTop; j++) {
			mMat[i][j] = mBeta1* mMat[i][j] + (1.0f - mBeta1)* dMat[i][j];
			vMat[i][j] = mBeta2* vMat[i][j] + (1.0f - mBeta2)* dMat[i][j] * dMat[i][j];
			float deltaW = (hat1* mMat[i][j]) / (sqrt(hat2* vMat[i][j]) + mEps);
			weight[i][j] -= mAlpha* deltaW;
		}
	}
}

void FcnLayer::Adam(int t, float beta1 = 0.9f, float beta2 = 0.999f, float alpha = 0.002f) {
	const float bias_factor = 2.0f;
	float mEps = 1e-8;

	float mBeta1 = beta1;
	float mBeta2 = beta2;
	float mAlpha = alpha;

	float hat1 = 1.0f / (1.0f - pow(mBeta1, t));
	float hat2 = 1.0f / (1.0f - pow(mBeta2, t));

	for (int i = 0; i < wBottom; i++) {
		mbVec[i] = mBeta1* mbVec[i] + (1.0f - mBeta1)*dbVec[i];
		vbVec[i] = mBeta2* vbVec[i] + (1.0f - mBeta2)* dbVec[i] * dbVec[i];
		float deltaB = (hat1* mbVec[i]) / (sqrt(hat2* vbVec[i]) + mEps);
		bias[i] -= bias_factor*mAlpha* deltaB;
		for (int j = 0; j < wTop; j++) {
			mMat[i][j] = mBeta1* mMat[i][j] + (1.0f - mBeta1)* dMat[i][j];
			vMat[i][j] = mBeta2* vMat[i][j] + (1.0f - mBeta2)* dMat[i][j] * dMat[i][j];
			float deltaW = (hat1* mMat[i][j]) / (sqrt(hat2* vMat[i][j]) + mEps);
			weight[i][j] -= mAlpha* deltaW;
		}
	}
}

FcnLayer::FcnLayer() {
	//cout << "构造layer" << endl;
	top = NULL;
	bottom = NULL;

	wTop = 0;
	wBottom = 0;
}

FcnLayer::FcnLayer(DataNode* t, DataNode* b) {

	//cout << "构造layer:"<<top->name<<"->"<<bottom->name << endl;
	top = t;
	bottom = b;

	wTop = top->width;
	wBottom = bottom->width;

	//bias = NULL;
}

void FcnLayer::Malloc(float* weightLoc, float* biasLoc,
	float* dMatLoc, float* dbVecLoc,
	float* mMatLoc, float* mbVecLoc,
	float* vMatLoc, float* vbVecLoc) {
	bias = biasLoc;

	weight = new float*[wBottom];
	for (int i = 0; i < wBottom; i++) {
		weight[i] = weightLoc + i * wTop;
	}

	dMat = new float*[wBottom]();
	mMat = new float*[wBottom]();
	vMat = new float*[wBottom]();
	//dMat[0] = new float[(wTop+1)*(wBottom+1)]();
	//mMat[0] = new float[(wTop+1)*(wBottom+1)]();
	//vMat[0] = new float[(wTop+1)*(wBottom+1)]();
	for (int i = 0; i < wBottom; i++) {
		//dMat[i] = dMat[0]+i*wTop+1;
		//mMat[i] = mMat[0]+i*wTop+1;
		//vMat[i] = vMat[0]+i*wTop+1;
		dMat[i] = dMatLoc + i*wTop;
		mMat[i] = mMatLoc + i*wTop;
		vMat[i] = vMatLoc + i*wTop;
		int a = 1;
	}

	//dbVec = new float[wBottom]();
	//mbVec = new float[wBottom]();
	//vbVec = new float[wBottom]();
	dbVec = dbVecLoc;
	mbVec = mbVecLoc;
	vbVec = vbVecLoc;

	//if (wTop == 4)
	//cout << &weight[0][0] << "\t" << &weight[0][0]+sizeof(float) << endl;
	//cout << &weight[0][0] << "\t" << &weight[0][1] << "\t" << &weight[0][2] << "\t" << &weight[0][3] << endl;
}


void FcnLayer::FreeMem() {
	//delete[] dMat[0];
	//delete[] mMat[0];
	//delete[] vMat[0];

	delete[] dMat;
	delete[] mMat;
	delete[] vMat;
	//delete[] dbVec;
	//delete[] mbVec;
	//delete[] vbVec;
}

FcnLayer::~FcnLayer() {
	//cout << "析构" << top->name << "->" << bottom->name << endl;
}

void FcnLayer::printDMat() {
	cout.width(8);
	cout << bottom->name << "->" << top->name << endl;
	for (int i = 0; i < wBottom; i++) {
		for (int j = 0; j < wTop; j++) {
			cout << dMat[i][j] << "\t";
		}
		cout << endl;
	}
	cout << "---------------------" << endl;
	for (int i = 0; i < wBottom; i++) {
		cout << dbVec[i] << "\t";
	}
	cout << endl;
}
void FcnLayer::printPara() {
	cout.width(8);
	cout << top->name << "->" << bottom->name << endl;
	for (int i = 0; i < wBottom; i++) {
		for (int j = 0; j < wTop; j++) {
			cout << weight[i][j] << "\t";
		}
		cout << endl;
	}
	cout << "---------------------" << endl;
	for (int i = 0; i < wBottom; i++) {
		cout << bias[i] << "\t";
	}
	cout << endl;
}

//----------------------------------------
//NETWORK
//----------------------------------------

float* Fcn::Evaluate(float* x) {
	for (int i = 0; i < wIn; i++) {
		nodes[0].aVec[i] = x[i];
	}
	for (int i = 0; i < depth; i++) {
		layers[i].Forward();
	}
	return nodes[depth].aVec;
}

void Fcn::ForwardCS() {

}

void Fcn::TrainCPU() {
	if (finishFlag) {
		for (int n = 0; n < maxEpochs; n++)
		{
			for (int k = 0; k < ceil(dataSize / batchSize) + 1; k++) {
				float error = 0;
				int inputShift = (k*batchSize) % dataSize;
//Clear Derivative
				for (int i = 0; i < depth; i++) {
					layers[i].ClearD();
				}
//Data Preparation: loss vector, inner counter(no. in this batch)
//					input:nodes[0].aVec, output groundtruth: nodes[depth].groundTruth
				float* lossVec = new float[wOut]();
				int noSample = 0;
//Actual Training
				for (int i = 0; i < batchSize; i++) {
					if ((i + inputShift) >= dataSize)
						break;
					for (int j = 0; j < wIn; j++) {
						nodes[0].aVec[j] = input[i + inputShift][j];
						//cout << i+inputShift << endl;
					}
					for (int j = 0; j < wOut; j++) {
						nodes[depth].groundTruth[j] = output[i + inputShift][j];
					}
//Forward: per layer
					//nodes[0].printValue();
					for (int j = 0; j < depth; j++) {
						layers[j].Forward();
						//layers[j].printPara();
						//nodes[j + 1].printValue();
					}
//Loss calculation: MSE
					for (int j = 0; j < wOut; j++) {
						lossVec[j] += (nodes[depth].aVec[j] - nodes[depth].groundTruth[j])* dActivateFunc(nodes[depth].zVec[j], layers[depth-1].activateType);
					}

					noSample++;
				}
//mini batch BP
				for (int j = 0; j < wOut; j++) {
						nodes[depth].dVec[j] = lossVec[j] / noSample;
				}
				//nodes[depth].printDiff();
				for (int j = depth - 1; j >= 0; j--) {
					layers[j].Backward();
					//nodes[j].printDiff();
				}
				for (int j = 0; j < depth; j++) {
					layers[j].AccuGrad();
					//layers[j].printDMat();
					layers[j].AdamFixed(2*k + 1, 0.00005f);
				}

				for (int i = 0; i < batchSize; i++) {
					for (int j = 0; j < depth; j++) {
						layers[j].Forward();
					}

					for (int j = 0; j < nodes[depth].width; j++) {
						error += (nodes[depth].aVec[j] - nodes[depth].groundTruth[j])* (nodes[depth].aVec[j] - nodes[depth].groundTruth[j])/wOut;
					}
				}
				error = error / noSample;
				error = sqrt(error);
				trainErrors.push_back(error);

				cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
				cout << "LOSS:\t" << fixed <<setprecision(6) << error;

				delete[] lossVec;
			}
//validation
			cout << endl;
			float error = 0;
			for (int ii = 0; ii < validSize; ii++) {
				for (int jj = 0; jj < validSize; jj++) {
					//nodes[0].aVec = testIn[ii];
					//nodes[depth].groundTruth = testOut[ii];
					for (int j = 0; j < wIn; j++) {
						nodes[0].aVec[j] = testIn[ii][j];
					}
					for (int j = 0; j < wOut; j++) {
						nodes[depth].groundTruth[j] = testOut[ii][j];
					}
				}
				for (int jj = 0; jj < depth; jj++) {
					layers[jj].Forward();
				}
				for (int jj = 0; jj < nodes[depth].width; jj++) {				
					error += (nodes[depth].aVec[jj] - nodes[depth].groundTruth[jj])* (nodes[depth].aVec[jj] - nodes[depth].groundTruth[jj])/wOut;
				}
			}
			error = error / validSize;
			error = sqrt(error);
			validErrors.push_back(error);
			cout << "VALID_LOSS:\t" << error << "\t" << n << endl;


			if (validErrors[validErrors.size() - 1] >= validErrors[validErrors.size() - 2]
				&& validErrors[validErrors.size() - 2] >= validErrors[validErrors.size() - 3]
				&& n > minEpochs) {
				for (int i = 0; i < matSize; i++) {
					matPt[i] = matMin[i];
				}

				cout << "------------------------------------------" << endl;
				for (int ii = 0; ii < validSize; ii++) {
					for (int j = 0; j < wIn; j++) {
						nodes[0].aVec[j] = testIn[ii][j];
					}
					for (int j = 0; j < wOut; j++) {
						nodes[depth].groundTruth[j] = testOut[ii][j];
					}
					for (int jj = 0; jj < depth; jj++) {
						layers[jj].Forward();
					}
					for (int jj = 0; jj < nodes[depth].width; jj++) {
						string channel[3] = { "R: ", "G: ", "B: " };
						cout << channel[jj] << nodes[depth].aVec[jj] << "\t" << nodes[depth].groundTruth[jj] << endl;
					}
				}
				cout << "------------------------------------------" << endl;
				cout << "Min Validation Loss: " << noMin << endl;

				break;
			}
			else if (validErrors[validErrors.size()-1] <= validErrors[validErrors.size() - 2]) {
				for (int i = 0; i < matSize; i++) {
					matMin[i] = matPt[i];			
				}
				noMin = n;
			}
		}
}
	else cout << "Network not finished." << endl;
}

void Fcn::TrainCPU_1epoch(int n, float beta1GUI = 0.9f, float beta2GUI = 0.999f, float alphaGUI = 0.0001f) {
	if (finishFlag) {
		cout << endl;
		float beta1 = beta1GUI;
		float beta2 = beta2GUI;
		float alpha = alphaGUI;

		for (int k = 0; k < ceil(dataSize / batchSize) + 1; k++) {
			float error = 0;
			int inputShift = (k*batchSize) % dataSize;
			//forward
			for (int i = 0; i < depth; i++) {
				layers[i].ClearD();
			}

			float* lossVec = new float[wOut]();
			int noSample = 0;
			for (int i = 0; i < batchSize; i++) {
				if ((i + inputShift) >= dataSize)
					break;
				for (int j = 0; j < wIn; j++) {
					nodes[0].aVec[j] = input[i + inputShift][j];
					//cout << i+inputShift << endl;
				}
				for (int j = 0; j < wOut; j++) {
					nodes[depth].groundTruth[j] = output[i + inputShift][j];
				}

				//nodes[0].printValue();
				for (int j = 0; j < depth; j++) {
					layers[j].Forward();
					//layers[j].printPara();
					//nodes[j + 1].printValue();
				}
				for (int j = 0; j < wOut; j++) {
					lossVec[j] += (nodes[depth].aVec[j] - nodes[depth].groundTruth[j])* dActivateFunc(nodes[depth].zVec[j], layers[depth-1].activateType);
				}
				noSample++;
			}

			//mini batch BP
			for (int j = 0; j < wOut; j++) {
				nodes[depth].dVec[j] = lossVec[j] / noSample;
			}
			//nodes[depth].printDiff();
			for (int j = depth - 1; j >= 0; j--) {
				layers[j].Backward();
				//nodes[j].printDiff();
			}
			for (int j = 0; j < depth; j++) {
				layers[j].AccuGrad();
				//layers[j].printDMat();
				layers[j].Adam(k + 1, beta1, beta2, alpha);
			}

			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < depth; j++) {
					layers[j].Forward();
				}

				for (int j = 0; j < nodes[depth].width; j++) {
					error += (nodes[depth].aVec[j] - nodes[depth].groundTruth[j])* (nodes[depth].aVec[j] - nodes[depth].groundTruth[j]) / wOut;
				}
			}
			error = error / noSample;
			error = sqrt(error);
			trainErrors.push_back(error);


			//cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
			//cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
			cout << "LOSS:\t" << fixed << setprecision(6) << error << endl;// << ", " << nodes[depth].aVec[0] - nodes[depth].groundTruth[0] << ", " << nodes[depth].aVec[1] - nodes[depth].groundTruth[1] << ", " << nodes[depth].aVec[2] - nodes[depth].groundTruth[2];

			delete[] lossVec;
		}

		for (int jj = 0; jj < depth; jj++) {
			layers[jj].printPara();
		}
		//validation
		cout << endl;
		float error = 0;
		for (int ii = 0; ii < validSize; ii++) {
			//nodes[0].aVec = testIn[ii];
			//nodes[depth].groundTruth = testOut[ii];
			for (int j = 0; j < wIn; j++) {
				nodes[0].aVec[j] = testIn[ii][j];
			}
			for (int j = 0; j < wOut; j++) {
				nodes[depth].groundTruth[j] = testOut[ii][j];
			}
			//for (int jj = 0; jj < validSize; jj++) {
			for (int jj = 0; jj < depth; jj++) {
				layers[jj].Forward();
			}
			for (int jj = 0; jj < nodes[depth].width; jj++) {
				error += (nodes[depth].aVec[jj] - nodes[depth].groundTruth[jj])* (nodes[depth].aVec[jj] - nodes[depth].groundTruth[jj]) / wOut;
			}
			//}
		}
		error = error / validSize;
		error = sqrt(error);
		validErrors.push_back(error);
		cout << "VALID_LOSS:\t" << error << "\t" << n << endl;


		if (validErrors[validErrors.size() - 1] >= validErrors[validErrors.size() - 2]
			&& validErrors[validErrors.size() - 2] >= validErrors[validErrors.size() - 3]
			&& n > minEpochs) {
			for (int i = 0; i < matSize; i++) {
				matPt[i] = matMin[i];
			}

			cout << "------------------------------------------" << endl;
			for (int ii = 0; ii < validSize; ii++) {
				for (int j = 0; j < wIn; j++) {
					nodes[0].aVec[j] = testIn[ii][j];
				}
				for (int j = 0; j < wOut; j++) {
					nodes[depth].groundTruth[j] = testOut[ii][j];
				}
				for (int jj = 0; jj < depth; jj++) {
					layers[jj].Forward();
				}
				for (int jj = 0; jj < nodes[depth].width; jj++) {
					string channel[3] = { "R: ", "G: ", "B: " };
					//cout << channel[jj] << nodes[depth].aVec[jj] << "\t" << nodes[depth].groundTruth[jj] << endl;
				}
			}
			cout << "------------------------------------------" << endl;
			cout << "Min Validation Loss: " << noMin << endl;

			return;
		}
		else if (validErrors[validErrors.size() - 1] <= validErrors[validErrors.size() - 2]) {
			for (int i = 0; i < matSize; i++) {
				matMin[i] = matPt[i];
			}
			noMin = n;
		}

	}
	else cout << "Network not finished." << endl;
}

bool Fcn::TrainCPU_1batch(int num_batch, int num_epoch, float beta1GUI = 0.9f, float beta2GUI = 0.999f, float alphaGUI = 0.0001f) {
	if (finishFlag) {
		float beta1 = beta1GUI;
		float beta2 = beta2GUI;
		float alpha = alphaGUI;

		bool firstBatch = (num_batch == 0);
		bool lastBatch = FALSE;

		if (firstBatch) {
			for (int i = 0; i < depth; i++) {
				layers[i].ClearD();
			}
		}
		//train
		{
			int inputShift = (num_batch * batchSize) % dataSize;

			//data preparation
			float* lossVec = new float[wOut]();
			int noSample = 0;
			float error = 0;
			for (int i = 0; i < batchSize; i++) {
				if ((i + inputShift) >= dataSize) {
					lastBatch = TRUE;  break;
				}
				for (int j = 0; j < wIn; j++) {
					nodes[0].aVec[j] = input[i + inputShift][j];
					//cout << i+inputShift << endl;
				}
				for (int j = 0; j < wOut; j++) {
					nodes[depth].groundTruth[j] = output[i + inputShift][j];
				}

				//forward
				for (int i = 0; i < depth; i++) {
					layers[i].ClearD();
				}
				//nodes[0].printValue();
				for (int j = 0; j < depth; j++) {
					layers[j].Forward();
					//layers[j].printPara();
					//nodes[j + 1].printValue();
				}
				for (int j = 0; j < wOut; j++) {
					lossVec[j] += (nodes[depth].aVec[j] - nodes[depth].groundTruth[j])* dActivateFunc(nodes[depth].zVec[j], layers[depth-1].activateType);
				}
				noSample++;
			}
			//if (firstBatch)		cout << endl;

			//mini batch BP
			for (int j = 0; j < wOut; j++) {
				nodes[depth].dVec[j] = lossVec[j] / noSample;
			}
			//nodes[depth].printDiff();
			for (int j = depth - 1; j >= 0; j--) {
				layers[j].Backward();
				//nodes[j].printDiff();
			}
			for (int j = 0; j < depth; j++) {
				layers[j].AccuGrad();
				//layers[j].printDMat();
				layers[j].Adam(num_batch + 1, beta1, beta2, alpha);//？？？
			}

			//Calculate Loss
			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < depth; j++) {
					layers[j].Forward();
				}

				for (int j = 0; j < nodes[depth].width; j++) {
					error += (nodes[depth].aVec[j] - nodes[depth].groundTruth[j])* (nodes[depth].aVec[j] - nodes[depth].groundTruth[j]) / wOut;
				}
			}
			error = error / noSample;
			error = sqrt(error);
			trainErrors.push_back(error);

			cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
			//cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
			cout << "LOSS:\t" << fixed << setprecision(6) << error;// << ", " << nodes[depth].aVec[0] - nodes[depth].groundTruth[0] << ", " << nodes[depth].aVec[1] - nodes[depth].groundTruth[1] << ", " << nodes[depth].aVec[2] - nodes[depth].groundTruth[2];

			delete[] lossVec;

			//for (int jj = 0; jj < depth; jj++) {
			//	layers[jj].printPara();
			//}
		}
		//validation
		if (lastBatch)
		{
			cout << endl;

			float error = 0;
			for (int i = 0; i < dataSize; i++) {
				for (int j = 0; j < wIn; j++) {
					nodes[0].aVec[j] = input[i][j];
				}
				for (int j = 0; j < wOut; j++) {
					nodes[depth].groundTruth[j] = output[i][j];
				}
				for (int j = 0; j < depth; j++) {
					layers[j].Forward();
				}
				for (int j = 0; j < nodes[depth].width; j++) {
					error += (nodes[depth].aVec[j] - nodes[depth].groundTruth[j])* (nodes[depth].aVec[j] - nodes[depth].groundTruth[j]) / wOut;
				}
			}
			error = error / dataSize;
			error = sqrt(error);
			trainErrors.push_back(error);

			//cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
			//cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
			cout << "TRAIN_LOSS:\t" << fixed << setprecision(6) << error << endl;// << ", " << nodes[depth].aVec[0] - nodes[depth].groundTruth[0] << ", " << nodes[depth].aVec[1] - nodes[depth].groundTruth[1] << ", " << nodes[depth].aVec[2] - nodes[depth].groundTruth[2];


			//Cout << endl;
			error = 0;
			for (int ii = 0; ii < validSize; ii++) {
				//nodes[0].aVec = testIn[ii];
				//nodes[depth].groundTruth = testOut[ii];
				for (int j = 0; j < wIn; j++) {
					nodes[0].aVec[j] = testIn[ii][j];
				}
				for (int j = 0; j < wOut; j++) {
					nodes[depth].groundTruth[j] = testOut[ii][j];
				}
				//for (int jj = 0; jj < validSize; jj++) {
				for (int jj = 0; jj < depth; jj++) {
					layers[jj].Forward();
				}
				for (int jj = 0; jj < nodes[depth].width; jj++) {
					error += (nodes[depth].aVec[jj] - nodes[depth].groundTruth[jj])* (nodes[depth].aVec[jj] - nodes[depth].groundTruth[jj]) / wOut;
				}
				//}
			}
			error = error / validSize;
			error = sqrt(error);
			validErrors.push_back(error);
			cout << "VALID_LOSS:\t" << error << "\t" << num_epoch << endl;
			cout << "------------------------------------------" << endl;


			if (validErrors[validErrors.size() - 1] >= validErrors[validErrors.size() - 2]
				&& validErrors[validErrors.size() - 2] >= validErrors[validErrors.size() - 3]
				&& num_epoch > minEpochs) {
				for (int i = 0; i < matSize; i++) {
					matPt[i] = matMin[i];
				}

				cout << "------------------------------------------" << endl;
				for (int ii = 0; ii < validSize; ii++) {
					for (int j = 0; j < wIn; j++) {
						nodes[0].aVec[j] = testIn[ii][j];
					}
					for (int j = 0; j < wOut; j++) {
						nodes[depth].groundTruth[j] = testOut[ii][j];
					}
					for (int jj = 0; jj < depth; jj++) {
						layers[jj].Forward();
					}
					for (int jj = 0; jj < nodes[depth].width; jj++) {
						string channel[3] = { "R: ", "G: ", "B: " };
						//cout << channel[jj] << nodes[depth].aVec[jj] << "\t" << nodes[depth].groundTruth[jj] << endl;
					}
				}
				cout << "------------------------------------------" << endl;
				cout << "Min Validation Loss: " << noMin << endl;

				return lastBatch;
			}
			else if (validErrors[validErrors.size() - 1] <= validErrors[validErrors.size() - 2]) {
				for (int i = 0; i < matSize; i++) {
					matMin[i] = matPt[i];
				}
				noMin = num_epoch;
			}
		}
		return lastBatch;
	}
	else{
		cout << "Network not finished." << endl;
		return FALSE;
	}
}


Fcn::Fcn(int wX, int wY) {
	wIn = wX;
	wOut = wY;

	depth = 1;
	DataNode Node0("input", wIn);
	nodes.push_back(Node0);
}

Fcn::Fcn(float** x, float** y, int wX, int wY, int size, int bSize) {
	batchSize = bSize;
	dataSize = size;
	wIn = wX;
	wOut = wY;

	input = new float*[dataSize];
	output = new float*[dataSize];
	for (int i = 0; i < dataSize; i++) {
		input[i] = new float[wIn];
		for (int j = 0; j < wIn; j++) {
			input[i][j] = x[i][j];
		}
		output[i] = new float[wOut];
		for (int j = 0; j < wOut; j++) {
			output[i][j] = y[i][j];
		}
	}

	depth = 1;
	DataNode Node0("input", wIn);
	nodes.push_back(Node0);
}

void Fcn::SetTrainData(float** x, float** y, int size, int bSize) {
	batchSize = bSize;
	dataSize = size;

	input = new float*[dataSize];
	output = new float*[dataSize];
	for (int i = 0; i < dataSize; i++) {
		input[i] = new float[wIn];
		for (int j = 0; j < wIn; j++) {
			input[i][j] = x[i][j];
		}
		output[i] = new float[wOut];
		for (int j = 0; j < wOut; j++) {
			output[i][j] = y[i][j];
		}
	}

	//for (int i = 0; i < batchSize; i++) {
	//	nodes[0].aVec = input[i];
	//	nodes[depth].groundTruth = output[i];
	//}
}

void Fcn::SetValidData(float** x, float** y, int size) {
	validSize = size;

	testIn = new float*[validSize];
	testOut = new float*[validSize];
	for (int i = 0; i < validSize; i++) {
		testIn[i] = new float[wIn];
		for (int j = 0; j < wIn; j++) {
			testIn[i][j] = x[i][j];
		}
		testOut[i] = new float[wOut];
		for (int j = 0; j < wOut; j++) {
			testOut[i][j] = y[i][j];
		}
	}
}

void Fcn::AddFCNlayer(int width) {
	string name = "H" + to_string(depth);
	DataNode Node(name, width);
	nodes.push_back(Node);
	//nodes.resize(depth + 2);
	//nodes.resize(depth + 1);

	depth++;
}

void Fcn::Finish(float* wLoc, float* bLoc) {
	DataNode Node("output", wOut);
	nodes.push_back(Node);
	//nodes.push_back(Node);

	int size_mat = 0;
	int size_vec = 0;
	for (int i = 0; i < depth; i++) {
		FcnLayer newLayer(&nodes[i], &nodes[i + 1]);
		layers.push_back(newLayer);

		size_mat += layers[i].wTop*layers[i].wBottom;
		size_vec += layers[i].wBottom;
	}

	dMatLoc = new float[size_mat]();
	mMatLoc = new float[size_mat]();
	vMatLoc = new float[size_mat]();
	dbVecLoc = new float[size_vec]();
	mbVecLoc = new float[size_vec]();
	vbVecLoc = new float[size_vec]();

	float* dMatPt = dMatLoc;
	float* mMatPt = mMatLoc;
	float* vMatPt = vMatLoc;
	float* dbVecPt = dbVecLoc;
	float* mbVecPt = mbVecLoc;
	float* vbVecPt = vbVecLoc;

	float* weightPt = wLoc;
	float* biasPt = bLoc;
	for (int i = 0; i < depth; i++) {
		layers[i].Malloc(weightPt, biasPt, dMatPt, dbVecPt, mMatPt, mbVecPt, vMatPt, vbVecPt);
		layers[i].InitData();

		weightPt += layers[i].wTop*layers[i].wBottom;// *sizeof(float);
		biasPt += layers[i].wBottom;// *sizeof(float);

		dMatPt += layers[i].wTop*layers[i].wBottom;// *sizeof(float);
		mMatPt += layers[i].wTop*layers[i].wBottom;// *sizeof(float);
		vMatPt += layers[i].wTop*layers[i].wBottom;// *sizeof(float);
		dbVecPt += layers[i].wBottom;// *sizeof(float);
		mbVecPt += layers[i].wBottom;// *sizeof(float);
		vbVecPt += layers[i].wBottom;// *sizeof(float);
	}

	matSize = size_mat;
	matPt = wLoc;
	matMin = new float[matSize];

	finishFlag = TRUE;
}

Fcn::~Fcn() {
	for (int i = 0; i < nodes.size(); i++) {
		nodes[i].FreeMem();
	}
	for (int i = 0; i < layers.size(); i++) {
		layers[i].FreeMem();
	}

	if (finishFlag) {
		delete[] dMatLoc;
		delete[] mMatLoc;
		delete[] vMatLoc;
		delete[] dbVecLoc;
		delete[] mbVecLoc;
		delete[] vbVecLoc;
	}

	for (int i = 0; i < dataSize; i++) {
		delete[] input[i];
		delete[] output[i];
	}
	delete[] input;
	delete[] output;
	delete[] matMin;
}
