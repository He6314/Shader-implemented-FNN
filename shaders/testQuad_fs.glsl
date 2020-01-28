#version 430

const int MAX_WIDTH = 30;
const int MAX_DEPTH = 30;
layout(std430, binding = 12) buffer ControlParas
{
	int depth;
	int weightSize;
};

layout(std430, binding = 13) buffer ControlWidth
{
	int width[];
};

layout(std430, binding = 14) buffer MatSSBO
{
	float mat[];
};

//===============================================================
layout(location = 99) uniform int outL;
out vec4 fragcolor;           
in vec2 tex_coord;

vec4 Eval(vec2 input);
      
void main(void)
{
	fragcolor = Eval(tex_coord);// * textureColor;
}

float activateFunc(float x) {
	//return (x > 0) ? x : 0;//relu
	//return (x > 0) ? x : (0.05*x);//leaky relu
	return tanh(x);//tanh 2.0 / (1.0 + exp(-2.0*x)) - 1.0;
};

vec4 Eval(vec2 input){
	vec4 color = vec4(1.0);

	float nodes[MAX_WIDTH][MAX_DEPTH];
	nodes[0][0] = input.x;
	nodes[0][1] = input.y;

	int wShift = 0;
	int bShift = 0;
	for(int n=1;n<depth+1;n++){
		int wTop = width[n-1];
		int wBottom = width[n];

		for(int i=0;i<wBottom;i++){
			nodes[n][i] = 0;
			for(int j=0;j<wTop;j++){
				int wLoc = wShift+i*wTop+j;
				float weight = mat[wLoc];
				nodes[n][i] += weight * nodes[n-1][j];
			}
			int bLoc = weightSize + bShift + i;
			float bias = mat[bLoc];
			nodes[n][i] += bias;
			if(n<depth)
			nodes[n][i] = activateFunc(nodes[n][i]);
		}
		wShift += width[n-1]*width[n];
		bShift += width[n];
	}
	
	color.x = nodes[outL][0] + 0.5f;
	color.y = nodes[outL][1] + 0.5f;
	color.z = nodes[outL][2] + 0.5f;

	return color;
}
