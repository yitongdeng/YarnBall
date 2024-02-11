#include "YarnBall.h"
#include <cuda.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace YarnBall {
	Sim::Sim(int numVerts) {
		if (numVerts < 3) throw std::runtime_error("Too little vertices");

		meta.numVerts = numVerts;
		meta.gravity = vec3(0, -9.8, 0);
		meta.h = maxH;
		meta.drag = 0.2;
		meta.damping = 1e-6;

		meta.radius = 1e-4;
		meta.barrierThickness = 8e-4;

		meta.kCollision = 1e-5;
		meta.detectionScaler = 2.f;
		meta.frictionCoeff = 0.1f;
		meta.time = 0.f;
		meta.collisionPeriod = 4;
		meta.numItr = 16;
		meta.hashTableSize = max(1024, numVerts * COLLISION_HASH_RATIO) + 17;

		// Initialize vertices
		verts = new Vertex[numVerts];
		for (size_t i = 0; i < numVerts; i++) {
			verts[i].invMass = verts[i].lRest = 1;
			verts[i].vel = vec3(0);
			verts[i].kBend = 5.f;
			verts[i].kStretch = 100.f;
			verts[i].connectionIndex = -1;
			verts[i].flags = (uint32_t)VertexFlags::hasNext;
		}
		verts[numVerts - 1].flags = 0;
	}

	Sim::~Sim() {
		delete[] verts;

		if (vertBuffer) delete vertBuffer;
		if (d_meta) {
			cudaFree(meta.d_dx);
			cudaFree(meta.d_lastVels);
			cudaFree(meta.d_hashTable);
			cudaFree(meta.d_numCols);
			cudaFree(meta.d_collisions);
			cudaFree(d_meta);
		}
		if (d_error) cudaFree(d_error);
	}

	void Sim::configure(float density) {
		const int numVerts = meta.numVerts;

		meta.maxSegLen = 0;

		// Init mass and orientation
		for (int i = 0; i < numVerts; i++) {
			auto& v = verts[i];

			// Fix flags
			if (i < numVerts - 1) {
				bool hasPrev = v.flags & (uint32_t)VertexFlags::hasNext;
				verts[i + 1].flags = (verts[i + 1].flags & ~(uint32_t)VertexFlags::hasPrev) | (hasPrev ? (uint32_t)VertexFlags::hasPrev : 0);

				// If the segment doesnt exist, then we fix the rotation
				if (!hasPrev) v.flags |= (uint32_t)VertexFlags::fixOrientation;
			}

			if (!(bool)(v.flags & (uint32_t)VertexFlags::hasPrev) && !(bool)(verts[i + 1].flags & (uint32_t)VertexFlags::hasNext))
				throw std::runtime_error("Dangling segment. Yarns must be atleast 2 segments long");

			v.lRest = 1.f / numVerts;
			v.q = Kit::Rotor::identity();
			v.qRest = vec4(0, 0, 0, 1);

			float mass = 0;
			if (v.flags & (uint32_t)VertexFlags::hasPrev)
				mass += verts[i - 1].lRest;

			if (v.flags & (uint32_t)VertexFlags::hasNext) {
				auto& v1 = verts[i + 1];
				vec3 seg0 = v1.pos - v.pos;
				v.lRest = length(seg0);
				if (v.lRest == 0 || !glm::isfinite(v.lRest))
					throw std::runtime_error("0 length segment");
				v.q = Kit::Rotor::fromTo(vec3(1, 0, 0), normalize(seg0));

				mass += v.lRest;
			}

			mass *= 0.5f * density;

			if (mass != 0)
				v.invMass *= 1 / mass;
			else
				v.invMass = 0;

			meta.maxSegLen = max(meta.maxSegLen, v.lRest);
		}

		// Init rest orientation
		for (int i = 0; i < numVerts - 1; i++) {
			auto& v0 = verts[i];
			auto& v1 = verts[i + 1];
			verts[i].qRest = (vec4)(v0.q.inverse() * v1.q);
		}

		// Mesh for rendering
		cylMesh = Kit::genCylMesh(6, 1, false);

		// Init meta
		cudaMalloc(&d_meta, sizeof(MetaData));
		cudaMalloc(&d_error, 2 * sizeof(int));
		cudaMalloc(&meta.d_dx, sizeof(vec3) * numVerts);
		cudaMalloc(&meta.d_lastVels, sizeof(vec3) * numVerts);
		cudaMemset(meta.d_lastVels, 0, sizeof(vec3) * numVerts);
		cudaMalloc(&meta.d_hashTable, sizeof(int) * meta.hashTableSize);
		cudaMalloc(&meta.d_numCols, sizeof(uint16_t) * numVerts);
		cudaMalloc(&meta.d_collisions, sizeof(Collision) * numVerts * MAX_COLLISIONS_PER_SEGMENT);

		vertBuffer = new Kitten::CudaComputeBuffer(sizeof(Vertex), numVerts);
		meta.d_verts = (Vertex*)vertBuffer->cudaPtr;

		uploadMeta();
		upload();
		cudaDeviceSynchronize();
	}

	void Sim::setKStretch(float kStretch) {
		if (!d_meta) throw std::runtime_error("No rest length. Must call configure()");

		// Multiplied by rest length to make energy density consistent.
		// Each segment has l * E energy, where E = C.k.C
		// The l is moved into the kStretch
		for (int i = 0; i < meta.numVerts; i++)
			verts[i].kStretch = kStretch * verts[i].lRest;
	}

	void Sim::setKBend(float kBend) {
		if (!d_meta) throw std::runtime_error("No rest length. Must call configure()");

		// Scaled by the 4 below
		kBend *= 4;

		// Divded by rest length to make energy density consistent.
		// Each segment has l * E energy, where E = C.k.C
		// The l is moved into the kBend, but we also cheated because the darboux vectors
		// in C should have been scaled by 2/l. So in total we end up dividing once.
		for (int i = 0; i < meta.numVerts; i++)
			verts[i].kBend = kBend / verts[i].lRest;
	}

	void Sim::uploadMeta() {
		cudaMemcpy(d_meta, &meta, sizeof(MetaData), cudaMemcpyHostToDevice);
	}

	void Sim::upload() {
		cudaMemcpy(meta.d_verts, verts, sizeof(Vertex) * meta.numVerts, cudaMemcpyHostToDevice);
	}

	void Sim::download() {
		cudaMemcpy(verts, meta.d_verts, sizeof(Vertex) * meta.numVerts, cudaMemcpyDeviceToHost);
	}

	__global__ void zeroVels(Vertex* verts, vec3* lastVels, int numVerts) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= numVerts) return;

		verts[tid].vel = vec3(0);
		lastVels[tid] = vec3(0);
	}

	void Sim::zeroVelocities() {
		zeroVels << <(meta.numVerts + 1023) / 1024, 1024 >> > (meta.d_verts, meta.d_lastVels, meta.numVerts);
		checkCudaErrors(cudaGetLastError());
	}
}