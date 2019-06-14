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

in vec3 pos_attrib;
in vec2 tex_coord_attrib;
in vec3 normal_attrib;

out vec2 tex_coord;  

void main(void)
{
   gl_Position = PVM*vec4(pos_attrib, 1.0);
   tex_coord = tex_coord_attrib;
}