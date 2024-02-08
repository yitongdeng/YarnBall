#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "../YarnBall.h"

namespace YarnBall {
	__global__ void buildTable(MetaData* data, float errorRadius2, int* errorReturn) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		auto verts = data->d_verts;
		auto dxs = data->d_dx;
		if (!(bool)(verts[tid].flags & (uint32_t)VertexFlags::hasNext)) return;

		// Get node pos and hash
		const vec3 p0 = verts[tid].pos + dxs[tid];
		const vec3 p1 = verts[tid + 1].pos + dxs[tid + 1];
		if (length2(p1 - p0) > errorRadius2)
			errorReturn[1] = Sim::WARNING_SEGMENT_STRETCH_EXCEEDS_DETECTION_SCALER;

		const vec3 pos = 0.5f * (p0 + p1);
		const ivec3 cell = Kitten::getCell(pos, data->colGridSize);
		const int hash = Kitten::getCellHash(cell);

		// Insert
		const int* table = data->d_hashTable;
		const int tSize = data->hashTableSize;
		int entry = (hash % tSize + tSize) % tSize;
		while (true) {
			if (!atomicCAS((int*)&table[entry], 0, (int)tid))
				break;
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
		const vec3 p0 = verts[tid].pos + dxs[tid];
		const vec3 p1 = verts[tid + 1].pos + dxs[tid + 1];
		const ivec2 cid(verts[tid].connectionIndex, verts[tid + 1].connectionIndex);
		const vec3 pos = 0.5f * (p0 + p1);
		const ivec3 cell = Kitten::getCell(pos, data->colGridSize);
		const int tSize = data->hashTableSize;
		const int* table = data->d_hashTable;

		bool hasLower = !(bool)(verts[tid].flags & (uint32_t)VertexFlags::hasPrev);

		float r2 = 2 * data->detectionRadius;
		r2 *= r2;
		float mr2 = 2 * data->radius;
		mr2 *= mr2;

		const float invb = 1 / data->barrierThickness;

		Collision col;
		int numCols = 0;
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
						col.oid = table[entry];
						if (col.oid == 0) break;

						// Discrete collision detection
						if (col.oid != tid && col.oid != tid + 1 && col.oid != tid - 1)			// Exempt neighboring segments
							if ((cid.x < 0 || (cid.x != col.oid && cid.x != col.oid + 1)) &&
								(cid.y < 0 || (cid.y != col.oid && cid.y != col.oid + 1))) {	// Exempt segments with special vertex connections
								vec3 op0 = verts[col.oid].pos + dxs[col.oid];
								vec3 op1 = verts[col.oid + 1].pos + dxs[col.oid + 1];

								col.uv = Kit::lineClosestPoints(p0, p1, op0, op1);
								// Remove depulicate collisions if there is a previous segment and the collision happens on the lower corner
								if ((hasLower || col.uv.x > 0) && (!(bool)(verts[col.oid].flags & (uint32_t)VertexFlags::hasPrev) || col.uv.y > 0)) {

									col.uv = clamp(col.uv, vec2(0), vec2(1));
									col.normal = mix(p0, p1, col.uv.x) - mix(op0, op1, col.uv.y);
									float d2 = Kit::length2(col.normal);

									if (d2 < r2) {
										if (d2 < mr2) // Report interpenetration
											errorReturn[1] = Sim::WARNING_SEGMENT_INTERPENETRATION;
										if (numCols == MAX_COLLISIONS_PER_SEGMENT) {
											errorReturn[0] = Sim::ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED;
											return;
										}

										// Prescale the normal by invb to simplify the math later on
										col.normal *= inversesqrt(d2) * invb;

										// Add entry to collision list
										// This is getting executed by the opposing segment in the same way. 
										// So theoretically, this thread only needs to add itself (pray to the floating point gods)
										collisions[tid + numVerts * numCols] = col;
										numCols++;
									}
								}
							}

						// Check next entry
						entry = (entry + 1) % tSize;
					}
				}

		data->d_numCols[tid] = numCols;
	}

	void Sim::detectCollisions() {
		// Rebuild hashmap
		cudaMemset(meta.d_hashTable, 0, sizeof(int) * meta.hashTableSize);
		buildTable << <(meta.numVerts + 127) / 128, 128 >> > (d_meta, meta.maxSegLen * meta.detectionScaler, d_error);

		// Build collision list
		buildCollisionList << <(meta.numVerts + 127) / 128, 128 >> > (d_meta, d_error);
	}
}