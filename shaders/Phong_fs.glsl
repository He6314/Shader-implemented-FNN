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

layout(location=0) uniform sampler2D diffuse_color;

layout(location=0) out vec4 fragColor;    
layout(location=1) out vec4 normalMap;    
layout(location=2) out vec4 viewMap;    
layout(location=3) out vec4 ldMap;  
layout(location=4) out vec4 objPosMap;
     
layout(location=0) in vec2 tex_coord;
layout(location=1) in vec4 World_Pos;
layout(location=2) in vec4 World_Normal;
layout(location=3) in vec4 Obj_Pos;

vec4 shadingPhong(vec4 normal, vec4 view, vec4 light, vec4 amb);
vec4 shadingGGX(vec3 position, vec3 normal, vec3 view, vec3 light);
vec4 shadingToon1(vec4 normal, vec4 view, vec4 light, vec4 amb);
#define PI 3.1415927

vec3 RGB2YUV(vec3 rgb);
vec3 YUV2RGB(vec3 yuv);
vec3 RGB2HSV(vec3 rgb);
vec3 HSV2RGB(vec3 hsv);

void main(void)
{   
	vec4 Amb = texture(diffuse_color, tex_coord);
	vec4 View = normalize(World_CamPos-World_Pos);
	vec4 Light = normalize(vec4(-3.0,0.0,5.0,1.0) - World_Pos);
		//normalize(World_Pos - vec4(3.0,1.0,1.0,1.0));
		//normalize(vec4(1.0,1.0,1.0,0.0));

	vec3 rgb = 
	//shadingToon1(World_Normal,View,Light, Amb).rgb;
	//shadingPhong(World_Normal,View,Light,Amb).rgb;
	shadingGGX(World_Pos.xyz, World_Normal.xyz, View.xyz, Light.xyz).rgb;


	fragColor = vec4(rgb,1.0);//vec4(RGB2YUV(rgb),1.0);//texture(diffuse_color, tex_coord);//vec4(tex_coord,0.0, 1.0);//

	normalMap = World_Normal;//vec4(RGB2HSV(rgb),1.0);//
	viewMap = View;//vec4(HSV2RGB(fragColor.xyz),1.0);//
	ldMap = Light;
	objPosMap = World_Pos; //Obj_Pos; //vec4(1.0,0.0,0.0,1.0);//
	//ldMap.z = 1.0;
	viewMap.w = tex_coord.x;
	ldMap.w = tex_coord.y;
	normalMap.w = 1.0;

//	if(dot(World_Normal,View)<0){
//		fragColor = vec4(0.0,0.3,0.3,0.5);
//		normalMap = vec4(0.3,0.0,0.3,0.5);
//	}
}

//============================
//Phong
//============================

vec4 shadingPhong(vec4 normal, vec4 view, vec4 light, vec4 amb){
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
    
    vec4 spec = specular * vec4(0.1);
    vec4 diff = diffuse * vec4(0.1);
	vec4 ambi = ambient * amb;

    return vec4(ambient+diffuse+specular);//ambi+diff+spec;
}

//==============================
//Toon shading
//==============================
vec4 shadingToon1(vec4 normal, vec4 view, vec4 light, vec4 amb){
    vec3 n = normal.xyz;
    vec3 l = light.xyz;
    vec3 r = reflect(-l,n);
    vec3 v = view.xyz;
    float Cp = 0.4;
    float d = 0.2;
    float cosi = max(dot(n,l),0.0);
    float wi = 0.6;
    float pn = 4.0;
    float coss = max(dot(v,r),0.0);
    
    float diffuse = Cp * cosi* (1.0-d);
    float ambient = Cp * d;
    float specular = wi*pow(coss,pn);
    float rfl = specular + diffuse + ambient;
    
    float thresh1 = 0.7;
    float thresh2 = 0.1;

    vec3 col1 = vec3(1.0,1.0,1.0);
    vec3 col2 = vec3(0.2,0.6,0.9);
    vec3 col3 = vec3(0.1,0.1,0.4);
    vec3 colL = vec3(0.0);
    
    vec3 col = 
        //step(thresh1, rfl)* col1 + step(rfl,thresh1)*(step(thresh2,rfl)*col2 + step(rfl,thresh2)*col3);
        smoothstep(thresh1-0.05, thresh1+0.05, rfl)* col1 + (1.0-smoothstep(thresh1-0.05, thresh1+0.05, rfl))*(smoothstep(thresh2-0.05,thresh2+0.05, rfl)*col2 + (1.0-smoothstep(thresh2-0.05,thresh2+0.05,rfl))*col3);
    col = step(0.25,dot(v,n))*col + step(dot(v,n),0.25)*colL;
    
    return vec4(col,1.0);
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

vec4 shadingGGX(vec3 p, vec3 normal, vec3 view, vec3 light)
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

//=======================================================================
//convert
//=======================================================================
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

vec3 RGB2HSV(vec3 rgb){
	float R = rgb.r;
	float G = rgb.g;
	float B = rgb.b;

	float maxCh = step(R,G)*G + step(G,R)*R;
	maxCh = step(maxCh, B)*B + step(B,maxCh)*maxCh;
	float minCh = step(R,G)*R + step(G,R)*G;
	minCh = step(minCh,B)*minCh + step(B,minCh)*B;

	float nume = step(R,G)*step(R,B)* (step(B,G)*(B-R+2.0*(maxCh-minCh)) + step(G,B)*(G-R)+4.0*(maxCh-minCh)) + 
					(1-step(R,G)*step(R,B))* (step(B,G)*(G-B) + step(G,B)*(G-B+6.0*(maxCh-minCh)));
	float H = nume/(maxCh-minCh)/6.0;
	float S = (maxCh-minCh)/max(maxCh,1e-8);
	float V = maxCh;

	return vec3(H,S,V);
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