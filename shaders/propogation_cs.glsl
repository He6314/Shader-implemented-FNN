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

layout(std430, binding = 7) buffer train{
	float trainData[];
};
layout(std430, binding = 8) buffer validation{
	float validData[];
};
layout(std430, binding = 9) buffer id_buffer{
	float ids[];
};

layout(location = 4) uniform float time;//debug
//===============================================================

float aNodes[MAX_WIDTH][MAX_DEPTH];
float zNodes[MAX_WIDTH][MAX_DEPTH];
float dNodes[MAX_WIDTH][MAX_DEPTH];
//float groundtruth[MAX_WIDTH];
float loss[MAX_WIDTH];
	
float activateFunc(float x) {
	//return (x > 0) ? x : 0;//relu
	//return (x > 0) ? x : (0.05*x);//leaky relu
	return tanh(x);//tanh 2.0 / (1.0 + exp(-2.0*x)) - 1.0;
};

float dActivateFunc(float x) {
	//return float(x>0);//drelu
	//return (x > 0) ? 1.0 : 0.05;//d leaku relu
	return 1 - tanh(x)*tanh(x);
};

void Forward(){
	uint index = uint((ids[0]+gl_GlobalInvocationID.x)*(width[0]+width[depth]+3));//uint(ids[gl_GlobalInvocationID.x]);///*(+3);
	for(int i=0;i<width[0];i++){
		aNodes[0][i] = trainData[index+i];
		ids[i+2] = trainData[index+i];
	}

	int wShift = 0;
	int bShift = weightSize;
	for(int n=1;n<depth+1;n++){ 
		int wTop = width[n-1];
		int wBottom = width[n];

		for(int i=0;i<wBottom;i++){
			aNodes[n][i] = 0;
			zNodes[n][i] = 0;
			for(int j=0;j<wTop;j++){
				int wLoc = wShift+i*wTop+j;
				float weight = mat[wLoc];
				zNodes[n][i] += weight * aNodes[n-1][j];
			}
			int bLoc = bShift + i;
			float bias = mat[bLoc];
			zNodes[n][i] += bias;
			//if(n<depth)
			aNodes[n][i] = activateFunc(zNodes[n][i]);
			//else
			//aNodes[n][i] = zNodes[n][i];
		}
		wShift += width[n-1]*width[n];
		bShift += width[n];
	}
	
	for(int i=0;i<width[depth];i++){
		float groundTruth = trainData[index+width[0]+i];
		dNodes[depth][i] = (aNodes[depth][i] - groundTruth)* dActivateFunc(zNodes[depth][i]);//aNodes[depth][i];//groundTruth;//i;//
		
		loss[i] = (aNodes[depth][i] - groundTruth)*(aNodes[depth][i] - groundTruth);
		//ids[13+i]= groundTruth;
		//ids[9+i] = aNodes[depth][i];
	}
}

void Backward(){
	int wShift = weightSize - width[depth-1]*width[depth];

	for(int n=depth-1;n>=0;n--){ 
		int wTop = width[n];
		int wBottom = width[n+1];

		for(int j=0;j<wTop;j++){
			dNodes[n][j] = 0;//?
			for(int i=0;i<wBottom;i++){
				int wLoc = wShift+i*wTop+j;
				float weight = mat[wLoc];
				dNodes[n][j] += dNodes[n+1][i] * weight;//;// * 应该对了
			}
			dNodes[n][j] *= dActivateFunc(zNodes[n][j]);
		}
		wShift -= width[n]*width[n-1];
	}
}

void AccuGrad(){
	int wShift = paraSize;
	int bShift = paraSize + weightSize;
	for(int n=1;n<depth+1;n++){ 
		int wTop = width[n-1];
		int wBottom = width[n];
		for(int i=0;i<wBottom;i++){
			for(int j=0;j<wTop;j++){
				int dW = wShift+i*wTop+j;
				mat[dW] = dNodes[n][i] * aNodes[n-1][j];
			}
			int dB = bShift + i;
			mat[dB] = dNodes[n][i];
		}
		wShift += width[n-1]*width[n];
		bShift += width[n];
	}
}

void main(void)
{
	Forward();
	Backward();
	AccuGrad();
	
	//ids[gl_GlobalInvocationID.x] = float(gl_GlobalInvocationID.x)/31.0;
	//ids[1] += 1.0/51319.0;//gl_GlobalInvocationID.x;//0.5;
	//ids[2] += gl_GlobalInvocationID.y;//1.0/51319.0;//0.5;
	//ids[3] += gl_GlobalInvocationID.z;//1.0/51319.0;//0.5;

	//mat[paraSize] = 2.0;
	ids[1] = sin(time);

	ids[2] = (loss[0]+loss[1]+loss[2])/3.0;
}

