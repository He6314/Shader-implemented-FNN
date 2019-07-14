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

layout(location = 10) uniform sampler2D buffer0;
layout(location = 11) uniform sampler2D buffer1;
layout(location = 12) uniform sampler2D buffer2;
layout(location = 13) uniform sampler2D buffer3;

out vec4 fragcolor;           
in vec2 tex_coord;
      
void main(void)
{   
	fragcolor = normalize(texture(buffer3, tex_coord));
	//fragcolor = vec4(1.0,0.0,0.0,1.0);
}
