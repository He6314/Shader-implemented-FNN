#version 430
#define LOCAL_WORKGROUP_SIZE_X 8
#define LOCAL_WORKGROUP_SIZE_Y 1
#define LOCAL_WORKGROUP_SIZE_Z 1

layout (local_size_x = 8, local_size_y = 1) in;

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

layout(std430, binding = 4) coherent buffer MatSSBO
{
	float mat[];
}try;

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


vec4 Eval(vec3 normal, vec3 view, vec3 light, vec2 tex_coord);
      
void main(void)
{
	//vec2 tex_coord = vec2(gl_GlobalInvocationID);
	//vec3 normal = texture(buffer1, tex_coord).xyz;
	//vec3 view = texture(buffer2, tex_coord).xyz;
	//vec3 light = texture(buffer3, tex_coord).xyz;
	//vec2 tex = texture(buffer2, tex_coord).ww;
	//tex.y = texture(buffer3, tex_coord).w;

	//vec4 textureColor = vec4(1.0);//texture(ambTexture, tex);
	
	//it won't be 0 in compute shader.
	//if(texture(buffer1, tex_coord).w!=0.0)
	//vec3 yuv = Eval(normal, view, light, tex).rgb;// * textureColor; //texture(buffer0, tex_coord);//
	//vec4 fragcolor = vec4(yuv,1.0);

	try.mat[0] = -10.0;//fragcolor[0];
	try.mat[1] = -10.0;//fragcolor[1];
	try.mat[2] = -10.0;//fragcolor[2];
	try.mat[3] = -10.0;//fragcolor[3];

	//imageStore(target, );
}

float activateFunc(float x) {
	//return (x > 0) ? x : 0;//relu
	//return (x > 0) ? x : (0.05*x);//leaky relu
	return tanh(x);//tanh 2.0 / (1.0 + exp(-2.0*x)) - 1.0;
};

