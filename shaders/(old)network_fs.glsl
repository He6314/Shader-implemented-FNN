#version 430

layout(std140, binding = 1) uniform Camera
{
	mat4 P;
	mat4 V;
	mat4 M;
	mat4 PV;
	mat4 PVM;
	mat4 Vinv;
	vec4 World_CamPos;
	vec2 Viewport;
};

const int MAX_WIDTH = 30;
const int MAX_DEPTH = 30;
layout(std430, binding = 2) buffer ControlParas
{
	int depth;
	int weightSize;
};

layout(std430, binding = 3) buffer ControlWidth
{
	int width[];
};

layout(std430, binding = 4) buffer MatSSBO
{
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

//===============================================================
uniform sampler2D ambTexture;

layout(location = 10) uniform sampler2D buffer0;
layout(location = 11) uniform sampler2D buffer1;
layout(location = 12) uniform sampler2D buffer2;
layout(location = 13) uniform sampler2D buffer3;

layout(location = 99) uniform int outL;

layout(location = 91) uniform int debugFlag;
layout(location = 92) uniform int depthIndex;
layout(location = 93) uniform int widthIndex1;
layout(location = 94) uniform int widthIndex2;
layout(location = 95) uniform int biasIndex;


out vec4 fragcolor;           
in vec2 tex_coord;

vec3 YUV2RGB(vec3 yuv){
	float Y = yuv.r;
	float U = yuv.g;
	float V = yuv.b;

	//float R = Y + 1.13983 * (V - 0.5);
	//float G = Y - 0.39465 * (U - 0.5) - 0.58060 * (V - 0.5);
	//float B = Y + 2.03211 * (V - 0.5);
	
	float R = Y + 1.402 * (V - 0.5);
	float G = Y - 0.3441 * (U - 0.5) - 0.7141 * (V - 0.5);
	float B = Y + 1.772 * (U - 0.5);

	return vec3(R,G,B);
}

vec4 Eval(vec3 normal, vec3 view, vec3 light, vec2 tex_coord);
      
void main(void)
{
	vec3 normal = texture(buffer1, tex_coord).xyz;
	vec3 view = texture(buffer2, tex_coord).xyz;
	vec3 light = texture(buffer3, tex_coord).xyz;
	vec2 tex = texture(buffer2, tex_coord).ww;
	tex.y = texture(buffer3, tex_coord).w;

	vec4 textureColor = vec4(1.0);//texture(ambTexture, tex);
	
	if(texture(buffer1, tex_coord).w!=0.0){
		vec3 yuv = Eval(normal, view, light, tex).rgb;// * textureColor; //texture(buffer0, tex_coord);//
		fragcolor = vec4(yuv,1.0);//YUV2RGB()
	}
	
	else
	{
		if(debugFlag!=1){
		 fragcolor = vec4(abs(aveIn[0])*10);}
		else{
			int wLoc = 0;
			for(int n=1;n<depthIndex+1;n++){ 
				wLoc += width[n-1]*width[n];}
			wLoc += widthIndex1 * width[depthIndex] + widthIndex2;
			float weight = mat[0];//[wLoc];		
			fragcolor = vec4(abs(weight));
		}
	}
}

float activateFunc(float x) {
	//return (x > 0) ? x : 0;//relu
	//return (x > 0) ? x : (0.2*x);//leaky relu
	return tanh(x);//tanh // 2.0 / (1.0 + exp(-2.0*x)) - 1.0; //another implementation
	//return 1.0 / (1.0 + exp(-x)); //sigmoid
};

vec4 Eval(vec3 normal, vec3 view, vec3 light, vec2 tex_coord){
	vec4 color = vec4(1.0);

	float nodes[MAX_WIDTH][MAX_DEPTH];
	nodes[0][0] = normal.x - aveIn[0];
	nodes[0][1] = normal.y - aveIn[1];
	nodes[0][2] = normal.z - aveIn[2];
	nodes[0][3] = view.x - aveIn[3];
	nodes[0][4] = view.y - aveIn[4];
	nodes[0][5] = view.z - aveIn[5];
	//nodes[0][7] = light.y;
	//nodes[0][8] = light.z;
	//nodes[0][6] = light.x;
	//nodes[0][9] = tex_coord.x;
	//nodes[0][10] = tex_coord.y;
	//nodes[0][0] = tex_coord.x;
	//nodes[0][1] = tex_coord.y;

	int wShift = 0;
	int bShift = 0;
	for(int n=1;n<depth+1;n++){ //depth+1?
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
		bShift += width[n]; //有点问题？
	}

	//color.x = nodes[outL][0];// + aveOut[0];//10.*abs(mat[299]);//
	//color.y = nodes[outL][1];// + aveOut[1];//10.*abs(mat[299]);//
	//color.z = nodes[outL][2];// + aveOut[2];//10.*abs(mat[299]);//
	
	color.x = nodes[outL][0] + aveOut[0];//10.*abs(mat[299]);//
	color.y = nodes[outL][1] + aveOut[1];//10.*abs(mat[299]);//
	color.z = nodes[outL][2] + aveOut[2];//10.*abs(mat[299]);//

	//color = vec4(vec3(nodes[outL][0]),1.0);//vec4(mat[15]+1.0f);
	return color;
}
