#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../YarnBall.h"

namespace YarnBall {
	__global__ void simpleSpringItr(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		const float h = data->h;
		auto verts = data->d_verts;
		auto dxs = data->d_dx;

		float invMass = verts[tid].invMass;
		if (invMass == 0) return;
		vec3 pos = verts[tid].pos;
		vec3 dx = dxs[tid];

		pos += dx;

		// Hessian H
		mat3 H = mat3(1 / (invMass * h * h));
		vec3 f = 1 / (h * h * invMass) * (data->d_vels[tid] - dx);

		// Simple springs
		float k = 1e4;
		int flags = verts[tid].flags;
		if (flags & (uint32_t)VertexFlags::hasPrev) {
			// Ignore race conditions for now
			vec3 p0 = verts[tid - 1].pos + dxs[tid - 1];
			vec3 d = pos - p0;
			float L = length(d);
			float c = L - verts[tid - 1].lRest;
			H += (mat3(c * L) + (1 - c / L) * Kit::abT(d, d)) * (k / (L * L));
			f -= (k * c / L) * d;
		}

		if (flags & (uint32_t)VertexFlags::hasNext) {
			// Ignore race conditions for now
			vec3 p0 = verts[tid + 1].pos + dxs[tid + 1];
			vec3 d = pos - p0;
			float L = length(d);
			float c = L - verts[tid].lRest;
			H += (mat3(c * L) + (1 - c / L) * Kit::abT(d, d)) * (k / (L * L));
			f -= (k * c / L) * d;
		}

		dx += inverse(H) * f;

		dxs[tid] = dx;
	}

	constexpr int BLOCK_SIZE = 256;
	void Sim::iterateSpring() {
		simpleSpringItr << <(meta.numVerts + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream >> > (d_meta);
		checkCudaErrors(cudaGetLastError());
	}
}