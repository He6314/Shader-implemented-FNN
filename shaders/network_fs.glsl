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
	int paraSize;
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
	int t;
	float mBeta1;
	float mBeta2;
	float mAlpha;
};
//===============================================================
layout(location = 10) uniform sampler2D buffer0;//debug only: ground truth
layout(location = 11) uniform sampler2D buffer1;
layout(location = 12) uniform sampler2D buffer2;
layout(location = 13) uniform sampler2D buffer3;
layout(location = 14) uniform sampler2D buffer4;

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

vec3 HSV2RGB(vec3 hsv){
	float H = hsv.r;
	float S = hsv.g;
	float V = hsv.b;

	float h = floor(H*6.0);
	float f = H*6.0 - h;
	float p = V * (1.0-S);
	float q = V * (1.0-f*S);
	float t = V * (1.0-(1.0-f)*S);

	vec3 c1 = vec3(V,t,p);
	vec3 c2 = vec3(q,V,p);
	vec3 c3 = vec3(p,V,t);
	vec3 c4 = vec3(p,q,V);
	vec3 c5 = vec3(t,p,V);
	vec3 c6 = vec3(V,p,q);

	return  step(h,0.9)*c1+step(0.9,h)*(
			step(h,1.9)*c2+step(1.9,h)*(
			step(h,2.9)*c3+step(2.9,h)*(
			step(h,3.9)*c4+step(3.9,h)*(
			step(h,4.9)*c5+step(4.9,h)*(
						c6)))));
}

vec4 Eval(vec3 normal, vec3 view, vec3 light, vec2 tex_coord);
      
void main(void)
{
	vec3 normal = texture(buffer1, tex_coord).xyz;
	vec3 obj = texture(buffer4, tex_coord).xyz;
	vec3 view = texture(buffer2, tex_coord).xyz;
	vec3 light = texture(buffer3, tex_coord).xyz;
	vec2 tex = texture(buffer2, tex_coord).ww;
	tex.y = texture(buffer3, tex_coord).w;

	vec4 textureColor = vec4(1.0);//texture(ambTexture, tex);
	
	if(texture(buffer1, tex_coord).w!=0.0){
		vec3 yuv = Eval(normal, view, light, tex).rgb;// * textureColor; //texture(buffer0, tex_coord);//
		fragcolor =vec4(yuv,1.0);//vec4(YUV2RGB(yuv),1.0);// texture(buffer4, tex_coord);//vec4(1.0);//
	}

	else
	{
	//Debug
	//int idx = int(tex_coord.x*64.0);
	fragcolor = vec4(0.0);
		//vec4(abs(mat[paraSize+2])*50.0);
		//vec4(abs(mat[2])*5.0);
		//vec4(abs(ids[idx]));
		//vec4(abs(ids[1]));
		//vec4(abs(ids[1]));
		//vec4(mat[paraSize]/5.0);
		//vec4(mat[0]);
		//vec4(abs(mBeta2));

//		if(debugFlag!=1){
//		 fragcolor = vec4(validData[150]+0.80);}//vec4(abs(aveIn[0])*10);}
//		else{
//			int wLoc = 0;
//			for(int n=1;n<depthIndex+1;n++){ 
//				wLoc += width[n-1]*width[n];}
//			wLoc += widthIndex1 * width[depthIndex] + widthIndex2;
//			float weight = mat[0];//[wLoc];		
//			fragcolor = vec4(abs(weight));
//		}
	}

	//fragcolor.g -= 1.0;
}

float activateFunc(float x) {
	//return (x > 0) ? x : 0;//relu
	//return (x > 0) ? x : (0.05*x);//leaky relu
	return tanh(x);//tanh 2.0 / (1.0 + exp(-2.0*x)) - 1.0;
};

vec4 Eval(vec3 normal, vec3 view, vec3 light, vec2 tex_coord){
	vec4 color = vec4(1.0);

	float nodes[MAX_WIDTH][MAX_DEPTH];

//	for(int i=0;i<width[0];i++){
//		int n = int(tex_coord.x/64.0);
//		nodes[0][i] = trainData[n*32 + i];
//	}
	nodes[0][0] = normal.x;// - aveIn[0];
	nodes[0][1] = normal.y;// - aveIn[1];
	nodes[0][2] = normal.z;// - aveIn[2];
	nodes[0][3] = view.x;// - aveIn[3];
	nodes[0][4] = view.y;// - aveIn[4];
	nodes[0][5] = view.z;// - aveIn[5];
//	nodes[0][7] = light.y;
//	nodes[0][8] = light.z;
//	nodes[0][6] = light.x;
//	nodes[0][9] = tex_coord.x;
//	nodes[0][10] = tex_coord.y;
//	nodes[0][3] = tex_coord.x;
//	nodes[0][4] = tex_coord.y;

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
			//if(n<depth)
			nodes[n][i] = activateFunc(nodes[n][i]);
		}
		wShift += width[n-1]*width[n];
		bShift += width[n]; //有点问题？
	}

	color.x = nodes[outL][0];// + aveOut[0];//10.*abs(mat[299]);//
	color.y = nodes[outL][1];// + aveOut[1];//10.*abs(mat[299]);//
	color.z = nodes[outL][2];// + aveOut[2];//10.*abs(mat[299]);//];

	//color = vec4(vec3(nodes[outL][0]),1.0);//vec4(mat[15]+1.0f);
	return color;
}
