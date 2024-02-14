#version 430 core

#include "kittenCommonVert.glsl"
#include "yarn.glsl"

out vec3 norm;
out vec3 wPos;

uniform int numVerts;
uniform float radius;

void main() {
	int flags = verts[gl_InstanceID].flags;
	if ((flags & 2) == 0) {
		wPos = norm = vec3(0);
		gl_Position = vec4(-1);
		return;
	}

	vec3 p0 = verts[(flags & 1) != 0 ? gl_InstanceID - 1 : gl_InstanceID].pos;
	vec3 p1 = verts[gl_InstanceID].pos;
	vec3 p2 = verts[gl_InstanceID + 1].pos;
	vec3 p3 = verts[(verts[gl_InstanceID + 1].flags & 2) != 0 ? gl_InstanceID + 2 : gl_InstanceID + 1].pos;

	vec3 pos = cmrSpline(p0, p1, p2, p3, vPos.y);
	vec3 tangent = normalize(cmrSplineTangent(p0, p1, p2, p3, vPos.y));

	vec3 d = normalize(p2 - p1);
	vec3 n = cross(d, vec3(-1, 1, 1));
	if (length2(n) < 0.0001) n = cross(d, vec3(1, -1, 1));

	// Use this to define the two normal basis
	n = normalize(cross(tangent, n));

	n = n * vPos.x - cross(tangent, n) * vPos.z;
	wPos = radius * n + pos;

	gl_Position = vpMat * vec4(wPos, 1);

	norm = mat3(modelMat_n) * n;
}