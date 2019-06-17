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

uniform sampler2D diffuse_color;

out vec4 fragcolor;           
in vec2 tex_coord;
      
void main(void)
{   
	fragcolor = texture(diffuse_color, tex_coord);
	//fragcolor = vec4(1.0,0.0,0.0,1.0);
}




















