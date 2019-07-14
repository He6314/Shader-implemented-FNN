#pragma once

#include <windows.h>
#include <iostream>
#include <string>
#include <random>
#include <math.h>
#include <vector>
#include "fcn.hpp"

class Optimizer {
private:
	double learningRate = 1e-3;

public:
	FcnLayer* cur;

	virtual void train();
};

class Adam : Optimizer {
private:
	const double bias_factor = 2.0f;
	double beta1 = 0.9f;
	double beta2 = 0.999f;
	double mAlpha = 0.0005f;
	double mEps = 1e-8;

	double learningRate = 0.00005;

public:
	FcnLayer* cur;
	void train();
};

void Adam::Train(int t) {
	const double bias_factor = 2.0f;
	double mBeta1 = 0.9f;
	double mBeta2 = 0.999f;
	double mAlpha = 0.0005f;
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