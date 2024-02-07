#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../YarnBall.h"

namespace YarnBall {
	__device__ inline vec4 inverseTorque(vec3 f, vec4 b) {
		float f2 = length2(f);
		float s = sqrt(f2) + length(b);
		float D = 1 / (f2 - s * s);
		b *= D;
		return normalize(mat4(
			s - f.x, -f.y, -f.z, 0,
			-f.y, s + f.x, 0, f.z,
			-f.z, 0, s + f.x, -f.y,
			0, f.z, -f.y, s - f.x
		) * b);
	}

	__global__ void cosseratItr(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		const float h = data->h;
		const float damping = data->damping / h;
		auto verts = data->d_verts;
		auto dxs = data->d_dx;

		vec3 segD;
		Vertex v0 = verts[tid];
		// Linear change
		if (v0.invMass != 0) {
			// vec3 y = v0.pos + h * v0.vel + h * h * data->gravity;
			vec3 dx = dxs[tid];

			v0.pos += dx;

			// Hessian H
			mat3 H = mat3(1 / (v0.invMass * h * h));
			// vel has to overwritten to contain y
			vec3 f = 1 / (h * h * v0.invMass) * (v0.vel - v0.pos);

			// Simple springs
			if (v0.flags & (uint32_t)VertexFlags::hasPrev) {
				vec3 p0 = verts[tid - 1].pos + dxs[tid - 1];
				float invl = 1 / verts[tid - 1].lRest;
				vec3 c = (v0.pos - p0) * invl - verts[tid - 1].q * vec3(1, 0, 0);

				float k = verts[tid - 1].stretchK * invl;
				float d = k * invl;
				f += -k * c - (damping * d) * dx;
				H += mat3((1 + damping) * d);
			}

			if (v0.flags & (uint32_t)VertexFlags::hasNext) {
				vec3 p1 = verts[tid + 1].pos + dxs[tid + 1];
				float invl = 1 / v0.lRest;
				segD = (p1 - v0.pos) * invl;
				vec3 c = segD - v0.q * vec3(1, 0, 0);

				float k = v0.stretchK * invl;
				float d = k * invl;
				f += k * c - (damping * d) * dx;
				H += mat3((1 + damping) * d);
			}

			dx += inverse(H) * f;

			dxs[tid] = dx;
		}

		// Update segment orientation
		// This is done assuming some very very large invMoment (i.e. no inertia so static equilibrium)
		if (!(bool)(v0.flags & (uint32_t)VertexFlags::fixOrientation) != 0 && (v0.flags & (uint32_t)VertexFlags::hasNext)) {
			vec4 b(0);

			if (v0.flags & (uint32_t)VertexFlags::hasPrev) {
				auto qRest = Kit::Rotor(verts[tid - 1].qRest);
				auto qq = verts[tid - 1].q;
				float s = dot((qq.inverse() * v0.q).v, qRest.v) > 0 ? 1 : -1;
				b -= (verts[tid - 1].bendK * s) * (qq * qRest).v;
			}

			if (verts[tid + 1].flags & (uint32_t)VertexFlags::hasNext) {
				auto qq = verts[tid + 1].q;
				float s = dot((v0.q.inverse() * qq).v, v0.qRest) > 0 ? 1 : -1;
				b -= (v0.bendK * s) * (verts[tid + 1].q * Kit::Rotor(v0.qRest).inverse()).v;
			}

			segD *= -2 * v0.stretchK;
			v0.q = inverseTorque(segD, b);
			verts[tid].q = v0.q;
		}
	}

	constexpr int BLOCK_SIZE = 256;
	void Sim::iterateCosserat() {
		cosseratItr << <(meta.numVerts + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_meta);
		checkCudaErrors(cudaGetLastError());
	}
}