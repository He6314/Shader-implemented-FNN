#version 430

in vec3 pos_attrib;
in vec2 tex_coord_attrib;

out vec2 tex_coord;  

const vec4 quad_pos[4] = vec4[] ( vec4(-1.0, 1.0, 0.0, 1.0), vec4(-1.0, -1.0, 0.0, 1.0), vec4( 1.0, 1.0, 0.0, 1.0), vec4( 1.0, -1.0, 0.0, 1.0) );
//const vec2 quad_tex[4] = vec2[] ( vec2(-1.0, 1.0), vec2(-1.0, -1.0), vec2( 1.0, 1.0), vec2( 1.0, -1.0) );
const vec2 quad_tex[4] = vec2[] ( vec2(-10.0, 10.0), vec2(-10.0, -10.0), vec2(10.0, 10.0), vec2( 10.0, -10.0) );

void main(void)
{
   gl_Position = quad_pos[gl_VertexID];
   tex_coord = quad_tex[gl_VertexID];
}