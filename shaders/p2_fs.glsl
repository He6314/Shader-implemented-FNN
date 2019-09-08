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
layout(std140, binding = 2) uniform ControlParas
{
	int depth;
	int weightSize;
	int width[MAX_DEPTH];
};

layout(std140, binding = 4) uniform AveVectors
{
	float aveIn[MAX_WIDTH];
	float aveOut[MAX_WIDTH];
};

layout(std430, binding = 3) buffer MatSSBO
{
	float mat[];
};

layout(location = 10) uniform sampler2D buffer0;
layout(location = 11) uniform sampler2D buffer1;
layout(location = 12) uniform sampler2D buffer2;
layout(location = 13) uniform sampler2D buffer3;

layout(location = 99) uniform int outL;

out vec4 fragcolor;           
in vec2 tex_coord;

vec4 Eval(vec3 normal, vec3 view, vec3 light, vec2 tex_coord);
      
void main(void)
{
	vec3 normal = texture(buffer1, tex_coord).xyz;
	vec3 view = texture(buffer2, tex_coord).xyz;
	vec3 light = texture(buffer3, tex_coord).xyz;
	vec2 tex = texture(buffer2, tex_coord).ww;
	tex.y = texture(buffer3, tex_coord).w;
	
	if(texture(buffer1, tex_coord).w!=0.0)
	fragcolor = Eval(normal, view, light, tex);
	
	else fragcolor = vec4(width[-1]);//10.*vec4(aveIn[0]);//vec4(0.3,0.5,0.5,0.0);
	//10.*vec4(mat[70]);
	//normalize(texture(buffer3, tex_coord));
	//fragcolor = vec4(1.0,1.0,0.0,1.0);

}

float activateFunc(float x) {
	//return (x > 0) ? x : 0;//relu
	//return (x > 0) ? x : (0.05*x);//leaky relu
	return tanh(x);//tanh 2.0 / (1.0 + exp(-2.0*x)) - 1.0;
};

vec4 Eval(vec3 normal, vec3 view, vec3 light, vec2 tex_coord){
	vec4 color = vec4(1.0);

	int widthT[6] = {6, 12, 12,  12,  12, 3};
	//int widthT[10] = {2, 8, 8, 8, 8, 8, 8, 8, 8, 3};

	float nodes[MAX_WIDTH][MAX_DEPTH];
	nodes[0][0] = normal.x;// - aveIn[1];
	nodes[0][1] = normal.y;// - aveIn[0];
	nodes[0][2] = normal.z;// - aveIn[2];
	nodes[0][3] = view.x;// - aveIn[3];
	nodes[0][4] = view.y;// - aveIn[4];
	nodes[0][5] = view.z;// - aveIn[5];
	//nodes[0][7] = light.y;
	//nodes[0][8] = light.z;
	//nodes[0][6] = light.x;
	//nodes[0][9] = tex_coord.x;
	//nodes[0][10] = tex_coord.y;
	//nodes[0][0] = tex_coord.x;
	//nodes[0][1] = tex_coord.y;

	int wShift = 0;
	int bShift = 0;
	for(int n=1;n<depth+1;n++){
		int wTop = widthT[n-1];
		int wBottom = widthT[n];

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
		wShift += widthT[n-1]*widthT[n];
		bShift += widthT[n];
	}

	color.x = nodes[outL][0];// + aveOut[0];//10.*abs(mat[299]);//
	color.y = nodes[outL][1];// + aveOut[1];//10.*abs(mat[299]);//
	color.z = nodes[outL][2];// + aveOut[2];//10.*abs(mat[299]);//

	return color;
}
