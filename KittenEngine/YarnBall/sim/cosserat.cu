#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../YarnBall.h"
#include "KittenEngine/includes/modules/SymMat.h"

namespace YarnBall {
	using Kit::hess3;

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

#define BLOCK_SIZE (256)
	__global__ void cosseratItr(MetaData* data) {
		const int tid = (int)(blockIdx.x * (BLOCK_SIZE - 1) + threadIdx.x) - 1;
		const int numVerts = data->numVerts;
		if (tid >= numVerts || tid < 0) return;

		const float h = data->h;
		const float damping = data->damping / h;
		const auto verts = data->d_verts;
		const auto dxs = data->d_dx;

		// Linear change
		Vertex v0 = verts[tid];

		// Hessian H
		hess3 H = hess3(1 / (v0.invMass * h * h));
		// vel has been overwritten to contain y - pos
		vec3 dx = dxs[tid];
		vec3 f = 1 / (h * h * v0.invMass) * (v0.vel - dx);

		// Special connections energy
		if (v0.connectionIndex >= 0) {
			constexpr float stiffness = 4e1;
			vec3 p0 = verts[v0.connectionIndex].pos;
			vec3 p0dx = dxs[v0.connectionIndex];
			f -= stiffness * ((v0.pos - p0) + (dx - p0dx) + damping * dx);
			H.diag += stiffness * (1 + damping);
		}

		// We need to store absolute position and position updates seperatly for floating point precision
		// If we added these together, the update could be small enough to be rounded out, causing stability issues
		vec3 p1, p1dx;
		float stepLimit = INFINITY;
		vec3 f2(0);
		hess3 H2(0);

		if (v0.flags & (uint32_t)VertexFlags::hasNext) {
			p1 = verts[tid + 1].pos;
			p1dx = dxs[tid + 1];
			stepLimit = data->d_maxStepSize[tid];

			// Cosserat stretching energy
			{
				float invl = 1 / v0.lRest;
				vec3 c = ((p1 - v0.pos) + (p1dx - dx)) * invl - v0.q * vec3(1, 0, 0);

				float k = v0.kStretch * invl;
				float d = k * invl;
				f += k * c - (damping * d) * dx;
				f2 += -k * c - (damping * d) * p1dx;
				d *= 1 + damping;
				H.diag += d;
				H2.diag += d;
			}

			const float fricK = data->kFriction;
			const float invb = 1 / data->barrierThickness;
			const float radius = 2 * data->radius;
			const float fricMu = data->frictionCoeff;
			const auto collisions = data->d_collisions;
			const float kCol = data->kCollision * invb;

			// Collision energy of this segment
			const int numCols = data->d_numCols[tid];
			const auto lastPos = data->d_lastPos;
			for (int i = 0; i < numCols; i++) {
				int colID = collisions[tid + i * numVerts];

				vec3 b0 = lastPos[colID];
				vec3 b1 = lastPos[colID + 1];
				vec3 db0 = dxs[colID];
				vec3 db1 = dxs[colID + 1];

				// Compute collision UV and normal
				vec2 uv = Kit::segmentClosestPoints(
					vec3(0), (p1 - v0.pos) + (p1dx - dx),
					(b0 - v0.pos) + (db0 - dx), (b1 - v0.pos) + (db1 - dx));
				if (!glm::isfinite(uv.x) || !glm::isfinite(uv.y))
					uv = vec2(0.5);

				vec3 dpos = mix(v0.pos, p1, uv.x) - mix(b0, b1, uv.y);
				vec3 ddpos = mix(dx, p1dx, uv.x) - mix(db0, db1, uv.y);
				vec3 normal = dpos + ddpos;
				float d = length(normal);
				normal /= d;

				uv.y = uv.x;
				uv.x = 1 - uv.x;

				// Compute penetration
				d = d - radius;
				d *= invb;
				if (d > 1) continue;	// Not touching
				d = max(d, 1e-3f);		// Clamp to some small value. This is a ratio of the barrier thickness.

				// IPC barrier energy
				float invd = 1 / d;
				float logd = log(d);

				float dH = (-3 + (2 + invd) * invd - 2 * logd) * kCol * invb;
				float ff = -(1 - d) * (d - 1 + 2 * d * logd) * invd * kCol;
				f += (ff * uv.x - damping * dH * uv.x * uv.x * dot(normal, dx)) * normal;
				f2 += (ff * uv.y - damping * dH * uv.y * uv.y * dot(normal, p1dx)) * normal;

				dH *= 1 + damping;
				hess3 op = hess3::outer(normal);
				H += op * (dH * uv.x * uv.x);
				H2 += op * (dH * uv.y * uv.y);

				// Friction
				vec3 u = ddpos - dot(normal, ddpos) * normal;
				float ul = length(u);
				if (ul > 0) {
					float f1 = glm::min(fricK, fricMu * ff / ul);

					op.diag -= 1;

					f -= f1 * uv.x * u;
					H -= op * (Kit::pow2(uv.x) * f1);

					f2 -= f1 * uv.y * u;
					H2 -= op * (Kit::pow2(uv.y) * f1);
				}
			}
		}

		__shared__ vec4 forces[BLOCK_SIZE];
		__shared__ hess3 hessians[BLOCK_SIZE];
		forces[threadIdx.x] = vec4(f2, stepLimit);
		hessians[threadIdx.x] = H2;

		__syncthreads();

		// No reason to keep thread 0 going anymore
		if (!threadIdx.x) return;

		if (v0.flags & (uint32_t)VertexFlags::hasPrev) {
			vec4 v = forces[threadIdx.x - 1];
			stepLimit = min(stepLimit, v.w);
			f += vec3(v);
			H += hessians[threadIdx.x - 1];
		}

		if (v0.invMass != 0) {
			// Local solve
			vec3 delta = data->accelerationRatio * (inverse((mat3)H) * f);
			dx += delta;

			float l = length(dx);
			if (l > stepLimit && l > 0) dx *= stepLimit / l;

			// Apply update
			dxs[tid] = dx;
		}
	}

	__global__ void quaternionItr(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		const int numVerts = data->numVerts;
		if (tid >= numVerts || tid < 0) return;

		const auto verts = data->d_verts;
		const auto dxs = data->d_dx;

		// Linear change
		Vertex v0 = verts[tid];

		// Update segment orientation
		// This is done assuming some very very large invMoment (i.e. no inertia so static equilibrium)
		if (!(bool)(v0.flags & (uint32_t)VertexFlags::fixOrientation) != 0 && (v0.flags & (uint32_t)VertexFlags::hasNext)) {
			vec3 dx = dxs[tid];
			vec3 p1 = verts[tid + 1].pos;
			vec3 p1dx = dxs[tid + 1];

			// All this is from an alternate derivation from forced-base hair interpolation.
			v0.pos = ((p1 - v0.pos) + (p1dx - dx)) / v0.lRest;
			v0.pos *= -2 * v0.kStretch;

			vec4 b(0);
			if (v0.flags & (uint32_t)VertexFlags::hasPrev) {
				auto qRest = Kit::Rotor(verts[tid - 1].qRest);
				auto qq = verts[tid - 1].q;
				float s = dot((qq.inverse() * v0.q).v, qRest.v) > 0 ? 1 : -1;
				b -= (verts[tid - 1].kBend * s) * (qq * qRest).v;
			}

			if (v0.flags & (uint32_t)VertexFlags::hasNextOrientation) {
				auto qq = verts[tid + 1].q;
				float s = dot((v0.q.inverse() * qq).v, v0.qRest) > 0 ? 1 : -1;
				b -= (v0.kBend * s) * (verts[tid + 1].q * Kit::Rotor(v0.qRest).inverse()).v;
			}

			v0.q = inverseTorque(v0.pos, b);
			verts[tid].q = v0.q;
		}
	}

	void Sim::iterateCosserat() {
		cosseratItr << <(meta.numVerts + BLOCK_SIZE - 2) / (BLOCK_SIZE - 1), BLOCK_SIZE, 0, stream >> > (d_meta);
		quaternionItr << <(meta.numVerts + 255) / 256, 256, 0, stream >> > (d_meta);
	}
}