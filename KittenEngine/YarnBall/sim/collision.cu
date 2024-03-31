#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "../YarnBall.h"

namespace YarnBall {
	__global__ void clearTable(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->hashTableSize) return;
		data->d_hashTable[tid] = INT_MAX;
	}

	__global__ void buildTable(MetaData* data, float errorRadius2, int* errorReturn) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		auto seg = data->d_lastSegments[tid];
		if (!glm::isfinite(seg.position.x)) return;

		if (length2(seg.delta) > errorRadius2)
			errorReturn[1] = Sim::WARNING_SEGMENT_STRETCH_EXCEEDS_DETECTION_SCALER;

		const ivec3 cell = Kitten::getCell(seg.position + 0.5f * seg.delta, data->colGridSize);
		const int hash = Kitten::getCellHash(cell);

		// Insert
		const int* table = data->d_hashTable;
		const int tSize = data->hashTableSize;
		int entry = (hash % tSize + tSize) % tSize;

		int id = tid;
		while (true) {
			int old = atomicMin((int*)&table[entry], id);
			if (old == INT_MAX) break;
			id = max(old, id);
			entry = (entry + 1) % tSize;
		}
	}

	__global__ void buildCollisionList(MetaData* data, int* errorReturn) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		const int numVerts = data->numVerts;
		if (tid >= numVerts) return;

		auto segs = data->d_lastSegments;
		auto s0 = segs[tid];
		if (!glm::isfinite(s0.position.x)) return;

		const ivec3 cell = Kitten::getCell(s0.position + 0.5f * s0.delta, data->colGridSize);

		const int tSize = data->hashTableSize;
		const int* table = data->d_hashTable;

		float r2 = 2 * data->detectionRadius;
		r2 *= r2;
		float mr2 = 2 * data->radius;
		mr2 *= mr2;

		auto nCols = data->d_numCols;
		const auto collisions = data->d_collisions;
		// Visit every cell in the 3x3x3 neighborhood
		for (ivec3 d(-1); d.x < 2; d.x++)
			for (d.y = -1; d.y < 2; d.y++)
				for (d.z = -1; d.z < 2; d.z++) {
					// Retrieve entries
					const ivec3 ncell = cell + d;
					const int hash = Kitten::getCellHash(ncell);
					int entry = (hash % tSize + tSize) % tSize;
					while (true) {
						Collision col;
						col.oid = table[entry];
						if (col.oid > tid) break;	// Let the higher thread handle this collision
						entry = (entry + 1) % tSize;

						// Discrete collision detection
						if (abs(col.oid - tid) > 2) {			// Exempt neighboring segments
							auto s1 = segs[col.oid];
							vec3 diff = s1.position - s0.position;
							col.uv = Kit::segmentClosestPoints(vec3(0), s0.delta, diff, diff + s1.delta);
							if (!glm::isfinite(col.uv.x) || !glm::isfinite(col.uv.y))
								col.uv = vec2(0.5);

							// Remove depulicate collisions if there is a previous segment and the collision happens on the lower corner
							col.normal = col.uv.x * s0.delta - (diff + col.uv.y * s1.delta);
							float d2 = Kit::length2(col.normal);

							if (d2 < r2) {
								if (d2 < mr2) // Report interpenetration
									errorReturn[1] = Sim::WARNING_SEGMENT_INTERPENETRATION;
								col.normal *= inversesqrt(d2);

								int numCols = atomicAdd(&nCols[tid], 1);
								if (numCols >= MAX_COLLISIONS_PER_SEGMENT)
									errorReturn[0] = Sim::ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED;
								else
									collisions[tid + numVerts * numCols] = col;

								int o = col.oid;
								col.oid = tid;
								col.normal *= -1;
								float tmp = col.uv.x;
								col.uv.x = col.uv.y;
								col.uv.y = tmp;

								numCols = atomicAdd(&nCols[o], 1);
								if (numCols >= MAX_COLLISIONS_PER_SEGMENT)
									errorReturn[0] = Sim::ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED;
								else
									collisions[o + numVerts * numCols] = col;
							}
						}
					}
				}

		/*
		int flag = verts[tid].flags;
		if (numCols > 0) flag |= (uint32_t)VertexFlags::colliding;
		else flag &= ~(uint32_t)VertexFlags::colliding;
		verts[tid].flags = flag;*/
	}

	void Sim::detectCollisions() {
		transferSegmentData();

		// Rebuild hashmap
		clearTable << <(meta.hashTableSize + 511) / 512, 512, 0, stream >> > (d_meta);
		buildTable << <(meta.numVerts + 127) / 128, 128, 0, stream >> > (d_meta, meta.maxSegLen * meta.detectionScaler, d_error);

		// Build collision list
		cudaMemsetAsync(meta.d_numCols, 0, sizeof(int) * meta.numVerts, stream);
		buildCollisionList << <(meta.numVerts + 127) / 128, 128, 0, stream >> > (d_meta, d_error);
	}

	__global__ void recomputeContactsKernel(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		const int numVerts = data->numVerts;
		if (tid >= numVerts) return;

		const auto verts = data->d_verts;
		const auto dxs = data->d_dx;
		if (!(bool)(verts[tid].flags & (uint32_t)VertexFlags::hasNext)) return;

		// Linear change
		vec3 p0 = verts[tid].pos;
		vec3 p0dx = dxs[tid];
		vec3 p1 = verts[tid + 1].pos;
		vec3 p1dx = dxs[tid + 1];
		Segment s0 = { p0 + p0dx, (p1 - p0) + (p1dx - p0dx) };

		// Collision energy of this segment
		const int numCols = data->d_numCols[tid];
		const auto collisions = data->d_collisions + tid;
		for (int i = 0; i < numCols; i++) {
			Collision col = collisions[i * numVerts];

			vec3 op0 = verts[col.oid].pos;
			vec3 op0dx = dxs[col.oid];
			vec3 op1 = verts[col.oid + 1].pos;
			vec3 op1dx = dxs[col.oid + 1];
			Segment s1 = { op0 + op0dx, (op1 - op0) + (op1dx - op0dx) };

			// Recompute contact data
			vec3 diff = s1.position - s0.position;
			col.uv = Kit::segmentClosestPoints(vec3(0), s0.delta, diff, diff + s1.delta);
			if (!glm::isfinite(col.uv.x) || !glm::isfinite(col.uv.y))
				col.uv = vec2(0.5);

			// Remove depulicate collisions if there is a previous segment and the collision happens on the lower corner
			col.normal = col.uv.x * s0.delta - (diff + col.uv.y * s1.delta);
			col.normal = normalize(col.normal);

			collisions[i * numVerts] = col;
		}
	}

	void Sim::recomputeContacts() {
		recomputeContactsKernel << <(meta.numVerts + 127) / 128, 128 >> > (d_meta);
	}

	__global__ void transferSegmentDataKernel(Vertex* verts, vec3* dxs, Segment* segment, int numVerts) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= numVerts) return;

		vec3 pos(NAN);
		vec3 delta(0);
		if (verts[tid].flags & (uint32_t)VertexFlags::hasNext) {
			pos = verts[tid].pos;
			vec3 dx = dxs[tid];
			delta = (verts[tid + 1].pos - pos) + (dxs[tid + 1] - dx);
			pos += dx;
		}
		segment[tid] = { pos, delta };
	}

	void Sim::transferSegmentData() {
		transferSegmentDataKernel << <(meta.numVerts + 127) / 128, 128 >> > (meta.d_verts, meta.d_dx, meta.d_lastSegments, meta.numVerts);
	}
}