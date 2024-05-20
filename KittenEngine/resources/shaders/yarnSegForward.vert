#version 430 core

#include "kittenCommonVert.glsl"
#include "yarn.glsl"

out vec3 col;
out vec3 norm;
out vec3 wPos;
out vec3 mPos;

uniform int numVerts;
uniform float radius;

void main() {
	Vertex n0 = verts[gl_InstanceID];
	Vertex n1 = verts[gl_InstanceID + 1];

	if ((n0.flags & 2) == 0) {
		col = wPos = mPos = vec3(0);
		gl_Position = vec4(-1);
		return;
	}

	vec3 axis = n1.pos - n0.pos;
	float l = length(axis);
	
	mat3 basis = orthoBasisY(axis / l);

	vec3 lPos = vPos;
	lPos *= vec3(radius, l, radius);

	wPos = (modelMat * vec4(basis * lPos + n0.pos, 1)).xyz;
	gl_Position = vpMat * vec4(wPos, 1);

	float k = l / n0.lRest - 1;
	col = vec3(1, 1, 1);
	col.xy *= 1 - 0.2 * (gl_InstanceID % 4);

	// if (n0.flags >= 8) col = vec3(1, 0, 0);

	norm = (modelMat * vec4(basis * (vec3(1, 0, 1) * vPos), 0)).xyz;

	vec4 q = qs[gl_InstanceID];

	// Rotation frame
	mPos = 4 * rotorMatrix(vec4(-q.xyz, q.w)) * basis * (vec3(1, l / radius, 1) * (vPos + vec3(0, -0.5, 0)));
}