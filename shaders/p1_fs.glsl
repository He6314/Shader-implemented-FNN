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

layout(location=0) out vec4 fragcolor;    
layout(location=1) out vec4 normalmap;
     
layout(location=0) in vec2 tex_coord;
layout(location=1) in vec4 World_Pos;
layout(location=2) in vec4 World_Normal;

vec4 shadingPhong(vec4 normal, vec4 view, vec4 light, vec4 amb);

void main(void)
{   
	vec4 Amb = 1.8*texture(diffuse_color, tex_coord);
	vec4 View = normalize(World_CamPos-World_Pos);
	vec4 Light = World_Pos - vec4(3.0,1.0,1.0,1.0);
		//normalize(vec4(1.0,1.0,1.0,0.0));
	fragcolor = shadingPhong(World_Normal,View,Light,Amb);

	normalmap = World_Normal;

	if(dot(World_Normal,View)<0){
		fragcolor = vec4(0.0,0.3,0.3,0.5);
		normalmap = vec4(0.3,0.0,0.3,0.5);
	}
}

vec4 shadingPhong(vec4 normal, vec4 view, vec4 light, vec4 amb)
{
    vec4 refle = reflect(light,normal);
    float Cp = 0.9;
    float d = 0.8;
    float cosi = max(dot(normal,light),0.0);
    float wi = 0.4;
    float n = 1.0;
    float coss = max(dot(view,refle),0.0);
    
    float diffuse = Cp * cosi* (1.0-d);
    float ambient = Cp * d;
    float specular = wi*pow(coss,n);
    
    vec4 spec = specular * vec4(0.7);
    vec4 diff = diffuse * vec4(0.7);
	vec4 ambi = ambient*amb;

    return ambi+diff+spec;
}



















