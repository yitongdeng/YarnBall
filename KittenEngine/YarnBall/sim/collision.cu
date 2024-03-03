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

		auto verts = data->d_verts;
		auto dxs = data->d_dx;
		if (!(bool)(verts[tid].flags & (uint32_t)VertexFlags::hasNext)) return;

		// Get node pos and hash
		vec3 p0 = verts[tid].pos;
		vec3 p1 = verts[tid + 1].pos;
		vec3 dx0 = dxs[tid];
		vec3 dx1 = dxs[tid + 1];

		if (length2((p1 - p0) + (dx1 - dx0)) > errorRadius2)
			errorReturn[1] = Sim::WARNING_SEGMENT_STRETCH_EXCEEDS_DETECTION_SCALER;

		const ivec3 cell = Kitten::getCell(0.5f * ((p0 + p1) + (dx0 + dx1)), data->colGridSize);
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

		auto verts = data->d_verts;
		auto dxs = data->d_dx;
		if (!(bool)(verts[tid].flags & (uint32_t)VertexFlags::hasNext)) return;

		// Get node pos and hash
		vec3 p0 = verts[tid].pos;
		vec3 dx0 = dxs[tid];
		vec3 p1 = verts[tid + 1].pos;
		vec3 dx1 = dxs[tid + 1];

		const ivec3 cell = Kitten::getCell(0.5f * ((p0 + p1) + (dx0 + dx1)), data->colGridSize);
		p1 = (p1 - p0) + (dx1 - dx0);

		const ivec2 cid(verts[tid].connectionIndex, verts[tid + 1].connectionIndex);
		const int tSize = data->hashTableSize;
		const int* table = data->d_hashTable;

		bool hasLower = !(bool)(verts[tid].flags & (uint32_t)VertexFlags::hasPrev);

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
						if (col.oid != tid && col.oid != tid + 1 && col.oid != tid - 1)			// Exempt neighboring segments
							if ((cid.x < 0 || (cid.x != col.oid && cid.x != col.oid + 1)) &&
								(cid.y < 0 || (cid.y != col.oid && cid.y != col.oid + 1))) {	// Exempt segments with special vertex connections
								vec3 op0 = (verts[col.oid].pos - p0) + (dxs[col.oid] - dx0);
								vec3 op1 = (verts[col.oid + 1].pos - p0) + (dxs[col.oid + 1] - dx0);

								col.uv = Kit::lineClosestPoints(vec3(0), p1, op0, op1);
								if (!glm::isfinite(col.uv.x) || !glm::isfinite(col.uv.y))
									col.uv = vec2(0.5);
								// Remove depulicate collisions if there is a previous segment and the collision happens on the lower corner
								// if ((hasLower || col.uv.x > 0) && (!(bool)(verts[col.oid].flags & (uint32_t)VertexFlags::hasPrev) || col.uv.y > 0)) {
								col.uv = clamp(col.uv, vec2(0), vec2(1));
								col.normal = col.uv.x * p1 - mix(op0, op1, col.uv.y);
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
								// }
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
		// Rebuild hashmap
		clearTable << <(meta.hashTableSize + 1023) / 1024, 1024, 0, stream >> > (d_meta);
		buildTable << <(meta.numVerts + 31) / 32, 32, 0, stream >> > (d_meta, meta.maxSegLen * meta.detectionScaler, d_error);

		// Build collision list
		cudaMemsetAsync(meta.d_numCols, 0, sizeof(int) * meta.numVerts, stream);
		buildCollisionList << <(meta.numVerts + 31) / 32, 32, 0, stream >> > (d_meta, d_error);
	}
}