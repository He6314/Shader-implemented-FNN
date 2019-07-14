#pragma once

#include <windows.h>
#include <iostream>
#include <string>
#include <random>
#include <math.h>
#include <vector>

using namespace std;

struct DataNode {
	//double zVec[50];
	//double aVec[50];
	//double dVec[50];

	//double groundTruth[50];

	double* zVec;
	double* aVec;
	double* dVec;

	double* groundTruth;

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
	//Don't allocate memory here, do it in the network for continuous and stable memory allocation.
	double** weight;
	double* bias;

	double** dMat;
	double* dbVec;
	double** mMat;
	double** vMat;
	double* mbVec;
	double* vbVec;

	//double weight[100][100];
	//double bias[100];
	//double dMat[100][100];
	//double dbVec[100];
	//double mMat[100][100];
	//double vMat[100][100];
	//double mbVec[100];
	//double vbVec[100];
	//==================================

	DataNode* top;
	DataNode* bottom;
	int wTop;
	int wBottom;

	void Forward();
	void Backward();
	void AccuGrad();
	void Adam(int t, double lr);

	void ClearD();
	void InitData();

	//for debug
	void printPara();
	void printDMat();

	void Malloc(double* weightLoc, double* biasLoc, double* dMatLoc, double* dbVecLoc, double* mMatLoc, double* mbVecLoc, double* vMatLoc, double* vbVecLoc);
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
	double** input;
	double** output;
	//double input[100][100];
	//double output[100][100];
	double** testIn;
	double** testOut;

	int dataSize;
	int batchSize;
	int validSize;

	double* dMatLoc;
	double* mMatLoc;
	double* vMatLoc;
	double* dbVecLoc;
	double* mbVecLoc;
	double* vbVecLoc;

	bool finishFlag = FALSE;

public:
	vector<double> trainErrors;
	vector<double> validErrors;
	int maxEpochs = 100;

	void TrainCPU();
	double* Evaluate(double* x); //not used

	void SetTrainData(double** x, double** y, int size, int bSize);
	void SetValidData(double** x, double** y, int size);
	void AddFCNlayer(int width);
	void Finish(double* weightLoc, double* biasLoc);

	int Depth() { return depth; }
	int NumDataNodes() { return depth + 1; }
	int NumHiddenLayer() { return depth - 1; }

	Fcn(int wX, int wY);
	Fcn(double** x, double** y, int wX, int wY, int dataSize, int batchSize);
	~Fcn();
};



///////////////////////////////////////////////////////////////////////
//Implementation
///////////////////////////////////////////////////////////////////////
//double tanh(double x) {
//	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
//}

double activateFunc(double x) {
	//return (x > 0) ? x : 0;//relu
	//return (x > 0) ? x : (0.05*x);//leaky relu
	return tanh(x);//tanh 2.0 / (1.0 + exp(-2.0*x)) - 1.0;
};
double dActivateFunc(double x) {
	//return double(x>0);//drelu
	//return (x > 0) ? 1.0 : 0.05;//d leaku relu
	return 1 - tanh(x)*tanh(x);
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

	zVec = new double[width];
	aVec = new double[width];
	dVec = new double[width];
	groundTruth = new double[width];
}

DataNode::DataNode(string newName, int w) {
	name = newName;
	//cout << "构造node:" << name << endl;
	width = w;

	zVec = new double[width];
	aVec = new double[width];
	dVec = new double[width];
	groundTruth = new double[width];
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
	//cout << &weight[0][0] << "\t" << &weight[0][0]+sizeof(double) << endl;
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
		bottom->aVec[i] = activateFunc(bottom->zVec[i]);
	}
}

void FcnLayer::Backward() {
	for (int j = 0; j < wTop; j++) {
		top->dVec[j] = 0;
		for (int i = 0; i < wBottom; i++) {
			top->dVec[j] += weight[i][j] * bottom->dVec[i];
		}
		top->dVec[j] *= dActivateFunc(top->zVec[j]);
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

void FcnLayer::Adam(int t, double lr) {
	const double bias_factor = 2.0f;
	double mBeta1 = 0.9f;
	double mBeta2 = 0.9f;//0.999f
	double mAlpha = 0.0001f;//0.0005f
	double mEps = 1e-8;

	double hat1 = 1.0f / (1.0f - pow(mBeta1, t));
	double hat2 = 1.0f / (1.0f - pow(mBeta2, t));

	for (int i = 0; i < wBottom; i++) {
		mbVec[i] = mBeta1* mbVec[i] + (1.0f - mBeta1)*dbVec[i];
		vbVec[i] = mBeta2* vbVec[i] + (1.0f - mBeta2)* dbVec[i] * dbVec[i];
		double deltaB = (hat1* mbVec[i]) / (sqrt(hat2* vbVec[i]) + mEps);
		bias[i] -= bias_factor*mAlpha* deltaB;
		for (int j = 0; j < wTop; j++) {
			mMat[i][j] = mBeta1* mMat[i][j] + (1.0f - mBeta1)* dMat[i][j];
			vMat[i][j] = mBeta2* vMat[i][j] + (1.0f - mBeta2)* dMat[i][j] * dMat[i][j];
			double deltaW = (hat1* mMat[i][j]) / (sqrt(hat2* vMat[i][j]) + mEps);
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

void FcnLayer::Malloc(double* weightLoc, double* biasLoc,
	double* dMatLoc, double* dbVecLoc,
	double* mMatLoc, double* mbVecLoc,
	double* vMatLoc, double* vbVecLoc) {
	bias = biasLoc;

	weight = new double*[wBottom];
	for (int i = 0; i < wBottom; i++) {
		weight[i] = weightLoc + i * wTop;
	}

	dMat = new double*[wBottom]();
	mMat = new double*[wBottom]();
	vMat = new double*[wBottom]();
	//dMat[0] = new double[(wTop+1)*(wBottom+1)]();
	//mMat[0] = new double[(wTop+1)*(wBottom+1)]();
	//vMat[0] = new double[(wTop+1)*(wBottom+1)]();
	for (int i = 0; i < wBottom; i++) {
		//dMat[i] = dMat[0]+i*wTop+1;
		//mMat[i] = mMat[0]+i*wTop+1;
		//vMat[i] = vMat[0]+i*wTop+1;
		dMat[i] = dMatLoc + i*wTop;
		mMat[i] = mMatLoc + i*wTop;
		vMat[i] = vMatLoc + i*wTop;
		int a = 1;
	}

	//dbVec = new double[wBottom]();
	//mbVec = new double[wBottom]();
	//vbVec = new double[wBottom]();
	dbVec = dbVecLoc;
	mbVec = mbVecLoc;
	vbVec = vbVecLoc;

	//if (wTop == 4)
	//cout << &weight[0][0] << "\t" << &weight[0][0]+sizeof(double) << endl;
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

double* Fcn::Evaluate(double* x) {
	for (int i = 0; i < wIn; i++) {
		nodes[0].aVec[i] = x[i];
	}
	for (int i = 0; i < depth; i++) {
		layers[i].Forward();
	}
	return nodes[depth].aVec;
}

void Fcn::TrainCPU() {
	if (finishFlag) {
		for (int n = 0; n < maxEpochs; n++)
		{
			for (int k = 0; k < ceil(dataSize / batchSize); k++) {
				double error = 0;
				double* lossVec = new double[wOut]();
				int inputShift = (k*batchSize) % dataSize;
//forward
				for (int i = 0; i < depth; i++) {
					layers[i].ClearD();
				}
				for (int i = 0; i < batchSize; i++) {
					for (int j = 0; j < wIn; j++) {
						nodes[0].aVec[j] = input[i + inputShift][j];
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
						lossVec[j] += (nodes[depth].aVec[j] - nodes[depth].groundTruth[j])* dActivateFunc(nodes[depth].zVec[i]);
					}
				}
//mini batch BP
				for (int j = 0; j < wOut; j++) {
					nodes[depth].dVec[j] = lossVec[j] / batchSize;
				}
				//nodes[depth].printDiff();
				for (int j = depth - 1; j >= 0; j--) {
					layers[j].Backward();
					//nodes[j].printDiff();
				}
				for (int j = 0; j < depth; j++) {
					layers[j].AccuGrad();
					//layers[j].printDMat();
					layers[j].Adam(2*k + 1, 0.00005f);
				}

				for (int i = 0; i < batchSize; i++) {
					for (int j = 0; j < depth; j++) {
						layers[j].Forward();
					}

					for (int j = 0; j < nodes[depth].width; j++) {
						error += 0.5*(nodes[depth].aVec[j] - nodes[depth].groundTruth[j])* (nodes[depth].aVec[j] - nodes[depth].groundTruth[j]);
					}
				}
				error = error / batchSize;
				trainErrors.push_back(error);
				//cout << "LOSS:\t" << error << endl;
			}
//validation
			double error = 0;
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
					string channel[3] = { "R: ", "G: ", "B: " };
//					cout << channel[jj] << nodes[depth].aVec[jj] << "\t" << nodes[depth].groundTruth[jj] << endl;
					error += 0.5*(nodes[depth].aVec[jj] - nodes[depth].groundTruth[jj])* (nodes[depth].aVec[jj] - nodes[depth].groundTruth[jj]);
				}
			}
			error = error / validSize;
			//validErrors.push_back(error);
			cout << "VALID_LOSS:\t" << error << "\t" << n << endl;
		}
	}
	else cout << "Network not finished." << endl;
}

Fcn::Fcn(int wX, int wY) {
	wIn = wX;
	wOut = wY;

	depth = 1;
	DataNode Node0("input", wIn);
	nodes.push_back(Node0);
}

Fcn::Fcn(double** x, double** y, int wX, int wY, int size, int bSize) {
	batchSize = bSize;
	dataSize = size;
	wIn = wX;
	wOut = wY;

	input = new double*[dataSize];
	output = new double*[dataSize];
	for (int i = 0; i < dataSize; i++) {
		input[i] = new double[wIn];
		for (int j = 0; j < wIn; j++) {
			input[i][j] = x[i][j];
		}
		output[i] = new double[wOut];
		for (int j = 0; j < wOut; j++) {
			output[i][j] = y[i][j];
		}
	}

	depth = 1;
	DataNode Node0("input", wIn);
	nodes.push_back(Node0);
}

void Fcn::SetTrainData(double** x, double** y, int size, int bSize) {
	batchSize = bSize;
	dataSize = size;

	input = new double*[dataSize];
	output = new double*[dataSize];
	for (int i = 0; i < dataSize; i++) {
		input[i] = new double[wIn];
		for (int j = 0; j < wIn; j++) {
			input[i][j] = x[i][j];
		}
		output[i] = new double[wOut];
		for (int j = 0; j < wOut; j++) {
			output[i][j] = y[i][j];
		}
	}

	//for (int i = 0; i < batchSize; i++) {
	//	nodes[0].aVec = input[i];
	//	nodes[depth].groundTruth = output[i];
	//}
}

void Fcn::SetValidData(double** x, double** y, int size) {
	validSize = size;

	testIn = new double*[validSize];
	testOut = new double*[validSize];
	for (int i = 0; i < validSize; i++) {
		testIn[i] = new double[wIn];
		for (int j = 0; j < wIn; j++) {
			testIn[i][j] = x[i][j];
		}
		testOut[i] = new double[wOut];
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

void Fcn::Finish(double* wLoc, double* bLoc) {
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

	dMatLoc = new double[size_mat]();
	mMatLoc = new double[size_mat]();
	vMatLoc = new double[size_mat]();
	dbVecLoc = new double[size_vec]();
	mbVecLoc = new double[size_vec]();
	vbVecLoc = new double[size_vec]();

	double* dMatPt = dMatLoc;
	double* mMatPt = mMatLoc;
	double* vMatPt = vMatLoc;
	double* dbVecPt = dbVecLoc;
	double* mbVecPt = mbVecLoc;
	double* vbVecPt = vbVecLoc;

	double* weightPt = wLoc;
	double* biasPt = bLoc;
	for (int i = 0; i < depth; i++) {
		layers[i].Malloc(weightPt, biasPt, dMatPt, dbVecPt, mMatPt, mbVecPt, vMatPt, vbVecPt);
		layers[i].InitData();

		weightPt += layers[i].wTop*layers[i].wBottom;// *sizeof(double);
		biasPt += layers[i].wBottom;// *sizeof(double);

		dMatPt += layers[i].wTop*layers[i].wBottom;// *sizeof(double);
		mMatPt += layers[i].wTop*layers[i].wBottom;// *sizeof(double);
		vMatPt += layers[i].wTop*layers[i].wBottom;// *sizeof(double);
		dbVecPt += layers[i].wBottom;// *sizeof(double);
		mbVecPt += layers[i].wBottom;// *sizeof(double);
		vbVecPt += layers[i].wBottom;// *sizeof(double);
	}

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
}
