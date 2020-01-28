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

layout(location=0) out vec4 fragColor;    
layout(location=1) out vec4 normalMap;    
layout(location=2) out vec4 viewMap;    
layout(location=3) out vec4 ldMap;
     
layout(location=0) in vec2 tex_coord;
layout(location=1) in vec4 World_Pos;
layout(location=2) in vec4 World_Normal;

vec4 shadingPhong(vec4 normal, vec4 view, vec4 light, vec4 amb);
vec4 shadingPBR(vec3 position, vec3 normal, vec3 view, vec3 light);
#define PI 3.1415927

vec3 RGB2YUV(vec3 rgb){
	float R = rgb.r;
	float G = rgb.g;
	float B = rgb.b;

	//float Y = 0.299 * R + 0.587 * G + 0.114 * B;
	//float U = -0.169 * R + 0.331 * G + 0.5 * B + 0.5;
	//float V = 0.5 * R + 0.419 * G + 0.081 * B + 0.5;

	float Y = 0.299 * R + 0.587 * G + 0.114 * B ;
	float U = -0.1687 * R - 0.3313 * G + 0.5 * B + 0.5;
	float V = 0.5 * R - 0.4187 * G - 0.0813 * B + 0.5;

	return vec3(Y,U,V);
}


void main(void)
{   
	vec4 Amb = 1.8*texture(diffuse_color, tex_coord);
	vec4 View = normalize(World_CamPos-World_Pos);
	vec4 Light = normalize(vec4(-3.0,0.0,5.0,1.0) - World_Pos);
		//normalize(World_Pos - vec4(3.0,1.0,1.0,1.0));
		//normalize(vec4(1.0,1.0,1.0,0.0));

		//shadingPhong(World_Normal,View,Light,Amb);
	vec3 rgb = shadingPBR(World_Pos.xyz, World_Normal.xyz, View.xyz, Light.xyz).rgb;

	fragColor = vec4(rgb,1.0);//RGB2YUV()

	normalMap = World_Normal;
	viewMap = View;
	ldMap = Light;
	//ldMap.z = 1.0;
	viewMap.w = tex_coord.x;
	ldMap.w = tex_coord.y;
	normalMap.w = 1.0;

//	if(dot(World_Normal,View)<0){
//		fragColor = vec4(0.0,0.3,0.3,0.5);
//		normalMap = vec4(0.3,0.0,0.3,0.5);
//	}
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
	vec4 ambi = ambient * amb;

    return vec4(ambient+diffuse+specular);//ambi+diff+spec;
}




//=======================================================================
//Will be written in another shader
//===============================================================================


vec3 fresnelSchlick(float cosTheta, vec3 F0){ return F0 + (1.0 - F0) * pow(1.0 - cosTheta,5.0); }

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
	
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
	
    return ggx1 * ggx2;
}

//================================================================

vec4 shadingPBR(vec3 p, vec3 normal, vec3 view, vec3 light)
{
    vec3 N = normalize(normal);  //normal
    vec3 V = normalize(view); // view direction
    
    //3: material parameters: metallic, roughness, AO
    float metallic = 1.0; 
    float roughness = 0.7;
    float ao = 1.0;

    //3 Physic para
    //3 Albedo para
    vec3 phyPara = vec3(1.0,0.3,0.7);
    vec3 albedo =  vec3(1.000, 0.766, 0.336);
    
    vec3 Lo = vec3(0.0);
    vec3 L = normalize(light);
    vec3 H = normalize(V+L);
    
    float attenDist = 0.5;//length(lightPos[i]-p);
    float attenuation = 1.0/(attenDist*attenDist);
    
    vec3 radiance = vec3(1.0)*attenuation;//white light
    
	vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, phyPara.x);
    vec3 F = fresnelSchlick(max(dot(H,V),0.0),F0);
    float NDF = DistributionGGX(N, H, phyPara.y);
    float G   = GeometrySmith(N, V, L, phyPara.y);
    
    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
    vec3 specular     = numerator / max(denominator, 0.001);
    
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - phyPara.x;
    
    float NdotL = max(dot(N, L), 0.0);
    Lo += (kD * albedo / PI  + specular) * radiance * NdotL;

    vec3 ambient = vec3(0.03) * albedo * phyPara.z;
    
    vec3 color = ambient + Lo;
    
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));
        
    return vec4(color,1.0);
}

