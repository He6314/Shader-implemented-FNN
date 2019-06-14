#pragma once

namespace UniformLoc
{
   const int P = 0;
   const int V = 1;
   const int M = 2;
   const int PVM = 3;
   const int Time = 4;
   const int DiffuseTex = 5;
   const int NormalTex = 6;
   const int BumpTex = 7;
   const int EnvTex = 8;
   const int Slider = 9;
   const int Mode = 10;

   const int pos_gbuffer = 11;
   const int normal_gbuffer = 12;
   const int diffuse_gbuffer = 13;

   const int ssaoTangentTex = 14;
   const int ssaoKernel = 15; //up to 64 locations
   const int ssaoTex = 79;
   const int ssaoNumSamples = 80;
   const int ssaoRadius = 81;
   const int ssaoBias = 82;
   const int ssaoTex0 = 83; //unfiltered

   const int deferredLightType = 84;
   const int depthTex = 85;
   const int lightPos = 86;
};

namespace AttribLoc
{
   const int Pos = 0;
   const int TexCoord = 1;
   const int Normal = 2;
   const int Tangent = 3;
   const int Bitangent = 4;
};

