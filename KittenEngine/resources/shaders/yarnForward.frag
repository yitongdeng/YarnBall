#version 430 core

#include "kittenCommonFrag.glsl"
#include "kittenLighting.glsl"

in vec3 col;
in vec3 norm;
in vec3 wPos;
in vec3 mPos;

out vec4 fragColor;

void main() {
	vec3 n = norm / length(norm);
	vec3 lDir = getLightDir(wPos);
	vec3 halfVec = 0.5 * (-lDir + getViewDir(wPos));

	float d = max(dot(-lDir, n), 0);

	fragColor = vec4(getLightCol(wPos) * d * matColor.xyz * col, 0);
}