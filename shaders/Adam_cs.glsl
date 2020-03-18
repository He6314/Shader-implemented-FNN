//NEW

#version 430
#define LOCAL_WORKGROUP_SIZE_X 1
#define LOCAL_WORKGROUP_SIZE_Y 1
#define LOCAL_WORKGROUP_SIZE_Z 1

layout (local_size_x = 1) in;

const int MAX_WIDTH = 30;
const int MAX_DEPTH = 30;
layout(std430, binding = 2) buffer ControlParas
{
	int depth;
	int weightSize;
	int paraSize;
};

layout(std430, binding = 3) buffer ControlWidth
{
	int width[];
};

layout(std430, binding = 4) buffer MatSSBO{
	float mat[];
};

layout(std430, binding = 5) buffer AveX
{
	float aveIn[];
};

layout(std430, binding = 6) buffer AveY
{
	float aveOut[];
};

layout(std430, binding = 7) buffer train{
	float trainData[];
};
layout(std430, binding = 8) buffer validation{
	float validData[];
};
layout(std430, binding = 9) buffer id_buffer{
	float ids[];
};

layout(std430, binding = 10) buffer hyperParas{
	int x;
};

//===============================================================
uniform sampler2D ambTexture;

layout(location = 10) uniform sampler2D buffer0;
layout(location = 11) uniform sampler2D buffer1;
layout(location = 12) uniform sampler2D buffer2;
layout(location = 13) uniform sampler2D buffer3;

layout (binding = 7, rgba16f) uniform image2D test_target;

float aNodes[MAX_WIDTH][MAX_DEPTH];
float zNodes[MAX_WIDTH][MAX_DEPTH];
float dNodes[MAX_WIDTH][MAX_DEPTH];
//float groundtruth[MAX_WIDTH];
float loss[MAX_WIDTH];

const float bias_factor = 2.0f;
float mEps = 1e-8;
float mBeta1 = 0.9;
float mBeta2 = 0.999;
float mAlpha = 0.002;

int t = 1;

float hat1 = 1.0f / (1.0f - pow(mBeta1, t));
float hat2 = 1.0f / (1.0f - pow(mBeta2, t));

int derivativeLoc = paraSize;
int momentLoc = 2*paraSize;
int velocityLoc = 3*paraSize;

void Adam(){
	int wShift = 0;
	int bShift = weightSize;
	for(int n=1;n<depth+1;n++) {
		int wTop = width[n-1];
		int wBottom = width[n];

		for (int i = 0; i < wBottom; i++) {
			int bLoc = bShift + i;
			int dbLoc = derivativeLoc + bShift + i;
			int mbLoc = momentLoc + bShift + i;
			int vbLoc = velocityLoc + bShift + i;

			mat[mbLoc] = mBeta1* mat[mbLoc] + (1.0f - mBeta1)*mat[dbLoc];
			mat[vbLoc] = mBeta2* mat[vbLoc] + (1.0f - mBeta2)* mat[dbLoc] * mat[dbLoc];
			float deltaB = (hat1* mat[mbLoc]) / (sqrt(hat2* mat[vbLoc]) + mEps);
			mat[bLoc] -= bias_factor * mAlpha * deltaB;

			for (int j = 0; j < wTop; j++) {
				int wLoc = wShift + i*wTop + j;
				int dwLoc = derivativeLoc + wShift + i*wTop + j;
				int mwLoc = momentLoc + wShift + i*wTop + j;
				int vwLoc = velocityLoc + wShift + i*wTop + j;
				mat[mwLoc] = mBeta1 * mat[mwLoc] + (1.0f - mBeta1)* mat[dwLoc];
				mat[vwLoc] = mBeta2* mat[vwLoc] + (1.0f - mBeta2)* mat[dwLoc] * mat[dwLoc];
				float deltaW = (hat1* mat[mwLoc]) / (sqrt(hat2 * mat[vwLoc]) + mEps);
				mat[wLoc] -= mAlpha* deltaW;
			}
		}
		wShift += width[n-1]*width[n];
		bShift += width[n];
	}
}


void main(void)
{
	Adam();
}

