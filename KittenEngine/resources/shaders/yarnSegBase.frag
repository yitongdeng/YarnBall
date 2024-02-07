#version 430 core

#include "kittenCommonFrag.glsl"
#include "kittenLighting.glsl"

out vec3 mPos;

out vec4 fragColor;

uniform float stripe;

void main() {
	vec4 mCol = matColor;
	mCol.yz *= -1 < mPos.x && mPos.x < 1 ? stripe : 1;
	mCol.xz *= -1 < mPos.y && mPos.y < 1 ? stripe : 1;
	mCol.xy *= -1 < mPos.z && mPos.z < 1 ? stripe : 1;

	fragColor = vec4(getLightCol(), 1) * mCol;
}