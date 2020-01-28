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

const vec4 quad_pos[4] = vec4[] ( vec4(-1.0, 1.0, 0.0, 1.0), vec4(-1.0, -1.0, 0.0, 1.0), vec4( 1.0, 1.0, 0.0, 1.0), vec4( 1.0, -1.0, 0.0, 1.0) );
const vec2 quad_tex[4] = vec2[] ( vec2(0.0, 1.0), vec2(0.0, 0.0), vec2( 1.0, 1.0), vec2( 1.0, 0.0) );

void main(void)
{
   gl_Position = quad_pos[gl_VertexID];
   tex_coord = quad_tex[gl_VertexID];
}