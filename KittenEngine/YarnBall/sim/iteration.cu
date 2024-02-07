#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../YarnBall.h"

namespace YarnBall {
	// Converts velocity to initial guess
	__global__ void initItr(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		const float h = data->h;
		auto verts = data->d_verts;
		auto lastVels = data->d_lastVels;

		const vec3 g = data->gravity;
		const vec3 vel = verts[tid].vel;
		const float invMass = verts[tid].invMass;

		vec3 dx = h * vel;
		vec3 lastVel = lastVels[tid];
		lastVels[tid] = vel;

		if (verts[tid].invMass != 0) {
			// Compute y (inertial + accel position)
			// Store it in vel (The actual vel is no longer needed)
			verts[tid].vel = verts[tid].pos + dx + (h * h) * g;

			// Compute initial guess
			vec3 a = (vel - lastVel) / data->lastH;
			float s = clamp(dot(a, g) / length2(g), 0.f, 1.f);
			dx += (h * h * s) * g;
		}

		data->d_dx[tid] = dx;
	}

	void Sim::startIterate() {
		initItr << <(meta.numVerts + 1023) / 1024, 1024 >> > (d_meta);
		checkCudaErrors(cudaGetLastError());
	}

	// Converts dx back to velocity and advects
	__global__ void endItr(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		const float h = data->h;
		const float invH = 1 / h;
		auto verts = data->d_verts;

		// Linear velocity
		vec3 dx = data->d_dx[tid];
		if (verts[tid].invMass != 0)
			verts[tid].vel = dx * invH * (1 - data->drag * h);
		verts[tid].pos += dx;
	}

	void Sim::endIterate() {
		endItr << <(meta.numVerts + 1023) / 1024, 1024 >> > (d_meta);
		checkCudaErrors(cudaGetLastError());
	}
}