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

	__global__ void buildCollisionList(MetaData* data, int maxCols, int* errorReturn) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		const int numVerts = data->numVerts;
		if (tid >= maxCols) return;

		ivec2 ids = data->d_boundColList[tid];

		auto segs = data->d_lastSegments;
		auto s0 = segs[ids.x];

		auto nCols = data->d_numCols;

		// Exempt self-collisions due to glueing
		if (s0.c0 == ids.y || s0.c1 == ids.y || s0.c0 == ids.y + 1 || s0.c1 == ids.y + 1) return;
		// Exempt neighboring segments
		if (abs(ids.y - ids.x) <= 2) return;

		auto s1 = segs[ids.y];
		// Discrete collision detection
		vec3 diff = s1.position - s0.position;

		vec2 uv = Kit::segmentClosestPoints(vec3(0), s0.delta, diff, diff + s1.delta);
		if (!glm::isfinite(uv.x) || !glm::isfinite(uv.y))
			uv = vec2(0.5);

		// Remove duplicate collisions if there is a previous segment and the collision happens on the lower corner
		vec3 normal = uv.x * s0.delta - (diff + uv.y * s1.delta);
		float d2 = Kit::length2(normal);

		float r = 2 * data->scaledDetectionRadius;
		float mr = 2 * data->radius;
		if (d2 < r * r) {
			if (d2 < mr * mr) // Report interpenetration
				errorReturn[1] = Sim::WARNING_SEGMENT_INTERPENETRATION;

			const auto collisions = data->d_collisions;
			int numCols = atomicAdd(&nCols[ids.x], 1);
			if (numCols >= MAX_COLLISIONS_PER_SEGMENT)
				errorReturn[0] = Sim::ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED;
			else
				collisions[ids.x + numVerts * numCols] = ids.y;

			numCols = atomicAdd(&nCols[ids.y], 1);
			if (numCols >= MAX_COLLISIONS_PER_SEGMENT)
				errorReturn[0] = Sim::ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED;
			else
				collisions[ids.y + numVerts * numCols] = ids.x;
		}
	}

	void Sim::detectCollisions() {
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

	__global__ void recomputeStepLimitKernel(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		const int numVerts = data->numVerts;
		if (tid >= numVerts) return;

		const auto verts = data->d_verts;
		const auto dxs = data->d_dx;
		if (!(bool)(verts[tid].flags & (uint32_t)VertexFlags::hasNext)) return;

		constexpr float SAFETY_MARGIN = 0.2f;

		// Linear change
		auto segs = data->d_lastSegments;
		auto s0 = segs[tid];

		// This is the maximum distance possible within with the AABB query is guaranteed to find a collision
		float minDist = data->detectionRadius * (data->detectionScaler - 1);

		// Collision energy of this segment
		const int numCols = data->d_numCols[tid];
		const auto collisions = data->d_collisions + tid;
		for (int i = 0; i < numCols; i++) {
			int oid = collisions[i * numVerts];
			Segment s1 = segs[oid];

			// Recompute contact data
			vec3 diff = s1.position - s0.position;
			vec2 uv = Kit::segmentClosestPoints(vec3(0), s0.delta, diff, diff + s1.delta);
			if (!glm::isfinite(uv.x) || !glm::isfinite(uv.y))
				uv = vec2(0.5);

			// Remove depulicate collisions if there is a previous segment and the collision happens on the lower corner
			vec3 normal = uv.x * s0.delta - (diff + uv.y * s1.delta);
			float l = length(normal);
			minDist = min(minDist, ((1 - SAFETY_MARGIN) * 0.5f) * l);
		}
		data->d_maxStepSize[tid] = minDist;
	}

	void Sim::recomputeStepLimit() {
		recomputeStepLimitKernel << <(meta.numVerts + 127) / 128, 128, 0, stream >> > (d_meta);
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