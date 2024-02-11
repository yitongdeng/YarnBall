#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../YarnBall.h"

namespace YarnBall {
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

	__global__ void cosseratItr(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		const int numVerts = data->numVerts;
		if (tid >= numVerts) return;

		const float h = data->h;
		const float damping = data->damping / h;
		const float kCol = data->kCollision;
		const auto verts = data->d_verts;
		const auto dxs = data->d_dx;
		const auto collisions = data->d_collisions;

		const float invb = 1 / data->barrierThickness;
		const float radius = 2 * data->radius;
		const float fricMu = data->frictionCoeff;

		Vertex v0 = verts[tid];
		// We need to store absolute position and position updates seperatly for floating point precision
		// If we added these together, the update could be small enough to be rounded out, causing stability issues
		vec3 p1, p1dx;
		if (v0.flags & (uint32_t)VertexFlags::hasNext) {
			p1 = verts[tid + 1].pos;
			p1dx = dxs[tid + 1];
		}

		// Linear change
		vec3 dx = dxs[tid];
		if (v0.invMass != 0) {
			const float fricE = 1e-3f * h;	// Friction epsilon dx theshold for static vs kinetic friction

			// Hessian H
			mat3 H = mat3(1 / (v0.invMass * h * h));
			// vel has been overwritten to contain y - pos
			vec3 f = 1 / (h * h * v0.invMass) * (v0.vel - dx);

			// Special connections energy
			if (v0.connectionIndex >= 0) {
				vec3 p0 = verts[v0.connectionIndex].pos;
				vec3 p0dx = dxs[v0.connectionIndex];
				f -= 4 * v0.kStretch * ((v0.pos - p0) + (dx - p0dx) + damping * dx);
				H += mat3(4 * (1 + damping) * v0.kStretch);
			}

			// Prev segment energy
			if (v0.flags & (uint32_t)VertexFlags::hasPrev) {
				vec3 p0 = verts[tid - 1].pos;
				vec3 p0dx = dxs[tid - 1];

				// Cosserat stretching energy
				{
					float invl = 1 / verts[tid - 1].lRest;
					vec3 c = ((v0.pos - p0) + (dx - p0dx)) * invl - verts[tid - 1].q * vec3(1, 0, 0);

					float k = verts[tid - 1].kStretch * invl;
					float d = k * invl;
					f += -k * c - (damping * d) * dx;
					H += mat3((1 + damping) * d);
				}

				// Collision energy of the previous segment
				const int numCols = data->d_numCols[tid - 1];
				for (int i = 0; i < numCols; i++) {
					Collision col = collisions[tid - 1 + i * numVerts];

					// Compute contact points
					vec3 dpos = mix(p0, v0.pos, col.uv.x) - mix(verts[col.oid].pos, verts[col.oid + 1].pos, col.uv.y);
					vec3 ddpos = mix(p0dx, dx, col.uv.x) - mix(dxs[col.oid], dxs[col.oid + 1], col.uv.y);

					// Compute penetration
					float d = (dot(col.normal, dpos + ddpos) - radius) * invb;
					if (d <= 0 || d > 1) continue;	// Either degenerate or not touching

					// IPC barrier energy
					float invd = 1 / d;
					float logd = log(d);
					float dH = (-3 + (2 + invd) * invd - 2 * logd) * Kit::pow2(col.uv.x) * kCol * invb * invb;
					float ff = -(1 - d) * (d - 1 + 2 * d * logd) * col.uv.x * invd * kCol * invb;
					f += (ff - dH * dot(col.normal, dx)) * col.normal;
					H += ((1 + damping) * dH) * glm::outerProduct(col.normal, col.normal);

					// Friction
					vec3 u = ddpos - dot(col.normal, ddpos) * col.normal;
					float ul = length(u);
					if (ul > 0) {
						float f1 = 1;
						if (ul < fricE) {
							f1 = ul / fricE;
							f1 = 2 * f1 - Kit::pow2(f1);
						}

						f1 *= fricMu * ff * col.uv.x / ul;
						f -= f1 * u;
						H += (col.uv.x * f1) * (mat3(1) - glm::outerProduct(col.normal, col.normal));
					}
				}
			}

			// Next segment energy
			if (v0.flags & (uint32_t)VertexFlags::hasNext) {
				// Cosserat stretching energy
				{
					float invl = 1 / v0.lRest;
					vec3 c = ((p1 - v0.pos) + (p1dx - dx)) * invl - v0.q * vec3(1, 0, 0);

					float k = v0.kStretch * invl;
					float d = k * invl;
					f += k * c - (damping * d) * dx;
					H += mat3((1 + damping) * d);
				}

				// Collision energy of this segment
				const int numCols = data->d_numCols[tid];
				for (int i = 0; i < numCols; i++) {
					Collision col = collisions[tid + i * numVerts];

					// Compute contact points
					vec3 dpos = mix(v0.pos, p1, col.uv.x) - mix(verts[col.oid].pos, verts[col.oid + 1].pos, col.uv.y);
					vec3 ddpos = mix(dx, p1dx, col.uv.x) - mix(dxs[col.oid], dxs[col.oid + 1], col.uv.y);

					// Compute penetration
					float d = (dot(col.normal, dpos + ddpos) - radius) * invb;
					if (d <= 0 || d > 1) continue;	// Either degenerate or not touching

					// IPC barrier energy
					float invd = 1 / d;
					float logd = log(d);
					float dH = (-3 + (2 + invd) * invd - 2 * logd) * Kit::pow2(1 - col.uv.x) * kCol * invb * invb;
					float ff = -(1 - d) * (d - 1 + 2 * d * logd) * (1 - col.uv.x) * invd * kCol * invb;
					f += (ff - dH * dot(col.normal, dx)) * col.normal;
					H += ((1 + damping) * dH) * glm::outerProduct(col.normal, col.normal);

					// Friction
					vec3 u = ddpos - dot(col.normal, ddpos) * col.normal;
					float ul = length(u);
					if (ul > 0) {
						float f1 = 1;
						if (ul < fricE) {
							f1 = ul / fricE;
							f1 = 2 * f1 - Kit::pow2(f1);
						}

						f1 *= fricMu * ff * (1 - col.uv.x) / ul;
						f -= f1 * u;
						H += ((1 - col.uv.x) * f1) * (mat3(1) - glm::outerProduct(col.normal, col.normal));
					}
				}
			}

			// Local solve and update
			dx += inverse(H) * f;
			dxs[tid] = dx;
		}

		// Update segment orientation
		// This is done assuming some very very large invMoment (i.e. no inertia so static equilibrium)
		if (!(bool)(v0.flags & (uint32_t)VertexFlags::fixOrientation) != 0 && (v0.flags & (uint32_t)VertexFlags::hasNext)) {
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

			if (verts[tid + 1].flags & (uint32_t)VertexFlags::hasNext) {
				auto qq = verts[tid + 1].q;
				float s = dot((v0.q.inverse() * qq).v, v0.qRest) > 0 ? 1 : -1;
				b -= (v0.kBend * s) * (verts[tid + 1].q * Kit::Rotor(v0.qRest).inverse()).v;
			}

			v0.q = inverseTorque(v0.pos, b);
			verts[tid].q = v0.q;
		}
	}

	constexpr int BLOCK_SIZE = 256;
	void Sim::iterateCosserat() {
		cosseratItr << <(meta.numVerts + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_meta);
		checkCudaErrors(cudaGetLastError());
	}
}