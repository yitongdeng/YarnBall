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
		meta.accelerationRatio = 1;

		meta.kCollision = 1e-5;
		meta.detectionScaler = 2.f;
		meta.frictionCoeff = 0.1f;
		meta.time = 0.f;
		meta.detectionPeriod = 1;
		meta.bvhRebuildPeriod = 8;
		meta.numItr = 8;

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
		if (stream) {
			cudaStreamSynchronize(stream);
			cudaStreamDestroy(stream);
		}

		if (vertBuffer) delete vertBuffer;
		if (d_meta) {
			cudaFree(meta.d_dx);
			cudaFree(meta.d_lastVels);
			cudaFree(meta.d_lastSegments);
			cudaFree(meta.d_numCols);
			cudaFree(meta.d_collisions);
			cudaFree(meta.d_bounds);
			cudaFree(meta.d_boundColList);
			cudaFree(d_meta);
		}
		if (d_error) cudaFree(d_error);
		if (stepGraph) cudaGraphExecDestroy(stepGraph);
		if (cylMesh) delete cylMesh;
		if (cylMeshHiRes) delete cylMeshHiRes;
	}

	void Sim::configure(float density) {
		const int numVerts = meta.numVerts;

		meta.maxSegLen = 0;
		meta.minSegLen = FLT_MAX;

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

				meta.maxSegLen = max(meta.maxSegLen, v.lRest);
				meta.minSegLen = min(meta.minSegLen, v.lRest);
			}

			mass *= 0.5f * density;

			if (mass != 0)
				v.invMass *= 1 / mass;
			else
				v.invMass = 0;
		}

		// Init rest orientation
		for (int i = 0; i < numVerts - 1; i++) {
			auto& v0 = verts[i];
			auto& v1 = verts[i + 1];
			verts[i].qRest = (vec4)(v0.q.inverse() * v1.q);
		}

		// Mesh for rendering
		cylMesh = Kit::genCylMesh(6, 1, false);
		cylMeshHiRes = Kit::genCylMesh(8, 6, false);

		// Init meta
		cudaMalloc(&d_meta, sizeof(MetaData));

		cudaMalloc(&d_error, 2 * sizeof(int));
		cudaMemset(d_error, 0, 2 * sizeof(int));

		cudaMalloc(&meta.d_dx, sizeof(vec3) * numVerts);

		cudaMalloc(&meta.d_lastVels, sizeof(vec3) * numVerts);
		cudaMemset(meta.d_lastVels, 0, sizeof(vec3) * numVerts);
		cudaMalloc(&meta.d_lastSegments, sizeof(Segment) * numVerts);

		cudaMalloc(&meta.d_numCols, sizeof(int) * numVerts);
		cudaMemset(meta.d_numCols, 0, sizeof(int) * meta.numVerts);
		cudaMalloc(&meta.d_collisions, sizeof(Collision) * numVerts * MAX_COLLISIONS_PER_SEGMENT);
		cudaMalloc(&meta.d_bounds, sizeof(Kit::LBVH::aabb) * numVerts);
		cudaMalloc(&meta.d_boundColList, sizeof(int) * numVerts * MAX_COLLISIONS_PER_SEGMENT);

		vertBuffer = new Kitten::CudaComputeBuffer(sizeof(Vertex), numVerts);
		meta.d_verts = (Vertex*)vertBuffer->cudaPtr;

		cudaDeviceSynchronize();
		cudaStreamCreate(&stream);
		uploadMeta();
		upload();
		checkCudaErrors(cudaGetLastError());
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
		meta.detectionRadius = meta.detectionScaler * (meta.radius + 0.5f * meta.barrierThickness);

		if (meta.minSegLen < 2 * meta.radius + meta.barrierThickness)
			throw std::runtime_error("Use thinner yarn or use longer segments. (Min seg length must be at least 2 * radius + barrierThickness");

		cudaMemcpyAsync(d_meta, &meta, sizeof(MetaData), cudaMemcpyHostToDevice, stream);
	}

	void Sim::upload() {
		cudaMemcpyAsync(meta.d_verts, verts, sizeof(Vertex) * meta.numVerts, cudaMemcpyHostToDevice, stream);
		cudaStreamSynchronize(stream);
	}

	void Sim::download() {
		cudaMemcpyAsync(verts, meta.d_verts, sizeof(Vertex) * meta.numVerts, cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
	}

	__global__ void zeroVels(Vertex* verts, vec3* lastVels, int numVerts) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= numVerts) return;

		verts[tid].vel = vec3(0);
		lastVels[tid] = vec3(0);
	}

	void Sim::zeroVelocities() {
		zeroVels << <(meta.numVerts + 1023) / 1024, 1024, 0, stream >> > (meta.d_verts, meta.d_lastVels, meta.numVerts);
		checkCudaErrors(cudaGetLastError());
	}

	void Sim::checkErrors() {
		checkCudaErrors(cudaGetLastError());

		int error[2];
		cudaMemcpyAsync(error, d_error, 2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);

		if (error[0] == ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED) {
			if (printErrors) fprintf(stderr, "ERROR: MAX_COLLISIONS_PER_SEGMENT exceeded. Current simulation state may be corrupted!\n");
			throw std::runtime_error("MAX_COLLISIONS_PER_SEGMENT exceeded");
		}
		else if (error[0] != ERROR_NONE) {
			if (printErrors) fprintf(stderr, "ERROR: Undescript error %d\n", error[0]);
			throw std::runtime_error("Indescript error");
		}

		if (printErrors)
			if (error[1] == WARNING_SEGMENT_STRETCH_EXCEEDS_DETECTION_SCALER)
				fprintf(stderr, "WARNING: Excessive segment stretching detected. Missed collisions possible due to insufficient detection radius.\n");
			else if (error[1] == WARNING_SEGMENT_INTERPENETRATION)
				fprintf(stderr, "WARNING: Interpenetration detection. This can be due to unstable contacts\n");
			else if (error[1] != ERROR_NONE)
				fprintf(stderr, "WARNING: Indescript warning %d\n", error[1]);

		if (error[0] != ERROR_NONE) lastErrorCode = error[0];
		if (error[1] != ERROR_NONE) lastWarningCode = error[1];

		// Reset errors
		if (error[0] != 0 || error[1] != 0)
			cudaMemsetAsync(d_error, 0, 2 * sizeof(int), stream);
	}
}