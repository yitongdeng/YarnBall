#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include <thrust/swap.h>

#include "../YarnBall.h"

namespace YarnBall {
	__global__ void buildAABBs(MetaData* data, int* errorReturn) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		auto seg = data->d_lastSegments[tid];
		Kit::LBVH::aabb aabb;

		if (glm::isfinite(seg.position.x)) {
			aabb.absorb(seg.position);
			aabb.absorb(seg.position + seg.delta);
			aabb.pad(data->scaledDetectionRadius);
		}

		data->d_bounds[tid] = aabb;
	}

	__device__ inline bool ignoreSeg(int i, int s) {
		return i - 2 <= s && s < i + 2;
	}

	__global__ void buildCollisionList(MetaData* data, int maxCols, int* errorReturn) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		const int numVerts = data->numVerts;
		if (tid >= maxCols) return;

		ivec2 ids = data->d_boundColList[tid];

		auto segs = data->d_lastSegments;
		auto s0 = segs[ids.x];
		auto s1 = segs[ids.y];

		float r2 = 2 * data->scaledDetectionRadius;
		r2 *= r2;
		float mr2 = 2 * data->radius;
		mr2 *= mr2;

		auto nCols = data->d_numCols;
		const auto collisions = data->d_collisions;

		// Exempt self-collisions due to glueing
		if (s0.c0 == ids.y || s0.c1 == ids.y || s0.c0 == ids.y + 1 || s0.c1 == ids.y + 1) return;
		// Exempt neighboring segments
		if (abs(ids.y - ids.x) <= 2) return;

		// Discrete collision detection
		vec3 diff = s1.position - s0.position;

		Collision col;
		col.uv = Kit::segmentClosestPoints(vec3(0), s0.delta, diff, diff + s1.delta);
		if (!glm::isfinite(col.uv.x) || !glm::isfinite(col.uv.y))
			col.uv = vec2(0.5);

		// Remove duplicate collisions if there is a previous segment and the collision happens on the lower corner
		col.normal = col.uv.x * s0.delta - (diff + col.uv.y * s1.delta);
		float d2 = Kit::length2(col.normal);

		if (d2 < r2) {
			if (d2 < mr2) // Report interpenetration
				errorReturn[1] = Sim::WARNING_SEGMENT_INTERPENETRATION;
			col.normal *= inversesqrt(d2);

			int numCols = atomicAdd(&nCols[ids.x], 1);
			if (numCols >= MAX_COLLISIONS_PER_SEGMENT)
				errorReturn[0] = Sim::ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED;
			else {
				col.oid = ids.y;
				collisions[ids.x + numVerts * numCols] = col;
			}

			numCols = atomicAdd(&nCols[ids.y], 1);
			if (numCols >= MAX_COLLISIONS_PER_SEGMENT)
				errorReturn[0] = Sim::ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED;
			else {
				col.oid = ids.x;
				col.normal *= -1;
				thrust::swap(col.uv.x, col.uv.y);
				collisions[ids.y + numVerts * numCols] = col;
			}
		}
	}

	void Sim::detectCollisions() {
		transferSegmentData();

		// Rebuild bvh
		buildAABBs << <(meta.numVerts + 255) / 256, 256 >> > (d_meta, d_error);
		if (lastBVHRebuild >= meta.bvhRebuildPeriod) {
			bvh.compute(meta.d_bounds, meta.numVerts);
			lastBVHRebuild = 0;
		}
		else {
			bvh.refit();
			lastBVHRebuild += meta.h * meta.detectionPeriod;
		}

		int numCols = bvh.query(meta.d_boundColList, meta.numVerts * MAX_COLLISIONS_PER_SEGMENT);

		// Build collision list
		cudaMemsetAsync(meta.d_numCols, 0, sizeof(int) * meta.numVerts, stream);
		buildCollisionList << <(numCols + 127) / 128, 128, 0, stream >> > (d_meta, numCols, d_error);
	}

	__global__ void recomputeContactsKernel(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		const int numVerts = data->numVerts;
		if (tid >= numVerts) return;

		const auto verts = data->d_verts;
		const auto dxs = data->d_dx;
		if (!(bool)(verts[tid].flags & (uint32_t)VertexFlags::hasNext)) return;

		constexpr float SAFETY_MARGIN = 0.2f;

		// Linear change
		vec3 p0 = verts[tid].pos;
		vec3 p0dx = dxs[tid];
		vec3 p1 = verts[tid + 1].pos;
		vec3 p1dx = dxs[tid + 1];
		Segment s0 = { p0 + p0dx, -1, (p1 - p0) + (p1dx - p0dx) };
		// This is the maximum distance possible within with the AABB query is guaranteed to find a collision
		float minDist = data->detectionRadius * (data->detectionScaler - 1);

		// Collision energy of this segment
		const int numCols = data->d_numCols[tid];
		const auto collisions = data->d_collisions + tid;
		for (int i = 0; i < numCols; i++) {
			Collision col = collisions[i * numVerts];

			vec3 op0 = verts[col.oid].pos;
			vec3 op0dx = dxs[col.oid];
			vec3 op1 = verts[col.oid + 1].pos;
			vec3 op1dx = dxs[col.oid + 1];
			Segment s1 = { op0 + op0dx, -1, (op1 - op0) + (op1dx - op0dx) };

			// Recompute contact data
			vec3 diff = s1.position - s0.position;
			col.uv = Kit::segmentClosestPoints(vec3(0), s0.delta, diff, diff + s1.delta);
			if (!glm::isfinite(col.uv.x) || !glm::isfinite(col.uv.y))
				col.uv = vec2(0.5);

			// Remove depulicate collisions if there is a previous segment and the collision happens on the lower corner
			col.normal = col.uv.x * s0.delta - (diff + col.uv.y * s1.delta);
			float l = length(col.normal);
			minDist = min(minDist, ((1 - SAFETY_MARGIN) * 0.5f) * l);
			col.normal *= 1 / l;

			collisions[i * numVerts] = col;
		}
		data->d_maxStepSize[tid] = minDist;
	}

	void Sim::recomputeContacts() {
		recomputeContactsKernel << <(meta.numVerts + 127) / 128, 128, 0, stream >> > (d_meta);
	}

	__global__ void transferSegmentDataKernel(Vertex* verts, vec3* dxs, Segment* segment, int numVerts) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= numVerts) return;

		vec3 pos(NAN);
		vec3 delta(0);
		ivec2 cid(-1);
		int flags = verts[tid].flags;
		if (flags & (uint32_t)VertexFlags::hasNext) {
			pos = verts[tid].pos;
			delta = verts[tid + 1].pos - pos;
			cid = ivec2(verts[tid].connectionIndex, verts[tid + 1].connectionIndex);
		}
		segment[tid] = { pos, cid.x, delta, cid.y };
	}

	void Sim::transferSegmentData() {
		transferSegmentDataKernel << <(meta.numVerts + 127) / 128, 128 >> > (meta.d_verts, meta.d_dx, meta.d_lastSegments, meta.numVerts);
	}
}