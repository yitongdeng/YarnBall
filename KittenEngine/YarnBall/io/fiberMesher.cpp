#include "../YarnBall.h"
#include "fastPRNG.h"

namespace YarnBall {
	inline float randf(fastPRNG::fastXS64& rng) {
		return glm::clamp((uint32_t)rng.xoroshiro128p() / (float)UINT32_MAX, 0.f, 1.f);
	}

	void Sim::exportFiberMesh(std::string path) {
		using namespace fastPRNG;
		using Kit::Rotor;

		const int64_t seed = 0x12345678;
		fastXS64 prng(seed);

		const float r = meta.radius + 0.5f * meta.barrierThickness;

		const int numFibers = 8;
		const float fiberFrizz = 0.2f * r;
		const float fiberTwist = 1200.f;

		const float fuzzDensity = 2e4;
		const float fuzzLength = 0.6f * r;

		const float fiberRadius = 0.6f * r / sqrt((float)numFibers);
		const float heightSegLen = 0.3f * r;
		const int fiberRadialSegs = 3;

		FILE* file = fopen(path.c_str(), "w");
		if (!file) throw std::runtime_error("Failed to open file for writing");

		download();

		int numVerts = 0;
		float curTwist = 0;
		for (size_t i = 0; i < meta.numVerts; i++) {
			auto v0 = verts[i];
			if (!(v0.flags & (uint32_t)VertexFlags::hasNext)) continue;

			auto v1 = verts[i + 1];

			Rotor q[3]{};
			q[0] = q[1] = q[2] = qs[i];

			auto va = v0;
			auto vb = v1;
			if (v0.flags & (uint32_t)VertexFlags::hasPrev) {
				va = verts[i - 1];
				q[0] = qs[i - 1];
			}
			if (v1.flags & (uint32_t)VertexFlags::hasNext) {
				vb = verts[i + 2];
				q[2] = qs[i + 1];
			}

			int numHeightSegs = glm::max((int)glm::ceil(v0.lRest / heightSegLen), 1);

			// Add fibers for this segment
			for (int fid = 0; fid < numFibers; fid++) {
				Rotor lastFrame;
				for (int hi = 0; hi <= numHeightSegs; hi++) {
					// CMR3 interpolation to get position
					float t = hi / (float)numHeightSegs;
					vec3 pos = Kit::cmrSpline(va.pos, v0.pos, v1.pos, vb.pos, t);
					Rotor frame = q[1];
					if (t < 0.5f) frame = normalize(mix(q[0].v, frame.v, t + 0.5f));
					else frame = normalize(mix(frame.v, q[2].v, t - 0.5f));

					float curTwistAngle = curTwist + fiberTwist * t * v0.lRest;
					curTwistAngle += fid / (float)numFibers * 2 * glm::pi<float>();
					vec3 twist(0, cos(curTwistAngle), sin(curTwistAngle));
					twist *= 0.9f * r - fiberRadius - fiberFrizz;

					if (hi && hi != numHeightSegs)
						twist += fiberFrizz * (vec3(randf(prng), randf(prng), randf(prng)) * 2.f - 1.f);

					// Write out a circle in obj format
					for (int ri = 0; ri < fiberRadialSegs; ri++) {
						float theta = 2 * glm::pi<float>() * ri / (float)fiberRadialSegs;
						vec3 lPos = vec3(0, cos(theta), sin(theta));
						vec3 wPos = pos + frame * (fiberRadius * lPos + twist);
						fprintf(file, "v %.8f %.8f %.8f\n", wPos.x, wPos.y, wPos.z);
						numVerts++;
					}

					// Write out faces
					if (hi > 0) {
						// Because the tubes can twist, we want to shift the triangle indices to match the previous frame
						float sinAngle = 2 * asinf((lastFrame.inverse() * frame).x);
						int twist = fiberRadialSegs - (int)floor((float)fiberRadialSegs * sinAngle / (2 * glm::pi<float>()));

						for (int ri = 0; ri < fiberRadialSegs; ri++) {
							int i0 = numVerts - 2 * fiberRadialSegs + ri + 1;
							int i1 = numVerts - 2 * fiberRadialSegs + (ri + 1) % fiberRadialSegs + 1;
							int i2 = numVerts - fiberRadialSegs + (ri + twist) % fiberRadialSegs + 1;
							int i3 = numVerts - fiberRadialSegs + (ri + twist + 1) % fiberRadialSegs + 1;

							fprintf(file, "f %d %d %d\n", i0, i1, i2);
							fprintf(file, "f %d %d %d\n", i3, i2, i1);
						}
					}
					lastFrame = frame;
				}
			}

			int numFuzz = (int)round(2 * randf(prng) * v0.lRest * fuzzDensity);
			for (int i = 0; i < numFuzz; i++) {
				// Randomly generate fuzz orientation and position
				float t = randf(prng);
				vec3 pos = Kit::cmrSpline(va.pos, v0.pos, v1.pos, vb.pos, t);
				Rotor frame = q[1];
				if (t < 0.5f) frame = normalize(mix(q[0].v, frame.v, t + 0.5f));
				else frame = normalize(mix(frame.v, q[2].v, t - 0.5f));

				frame = frame * Kit::Rotor(normalize(vec4(randf(prng), randf(prng), randf(prng), randf(prng)) * 2.f - 1.f));

				pos += 0.7f * r * (vec3(randf(prng), randf(prng), randf(prng)) * 2.f - 1.f);

				for (int hi = 0; hi <= 1; hi++) {
					float t = hi;

					// Write out a circle in obj format
					for (int ri = 0; ri < fiberRadialSegs; ri++) {
						float theta = 2 * glm::pi<float>() * ri / (float)fiberRadialSegs;
						vec3 lPos = vec3(fuzzLength * t, 0.4f * fiberRadius * cos(theta), 0.4f * fiberRadius * sin(theta));
						vec3 wPos = pos + frame * lPos;
						fprintf(file, "v %.8f %.8f %.8f\n", wPos.x, wPos.y, wPos.z);
						numVerts++;
					}

					// Write out faces
					if (hi > 0) {
						// Because the tubes can twist, we want to shift the triangle indices to match the previous frame
						for (int ri = 0; ri < fiberRadialSegs; ri++) {
							int i0 = numVerts - 2 * fiberRadialSegs + ri + 1;
							int i1 = numVerts - 2 * fiberRadialSegs + (ri + 1) % fiberRadialSegs + 1;
							int i2 = numVerts - fiberRadialSegs + ri % fiberRadialSegs + 1;
							int i3 = numVerts - fiberRadialSegs + (ri + 1) % fiberRadialSegs + 1;

							fprintf(file, "f %d %d %d\n", i0, i1, i2);
							fprintf(file, "f %d %d %d\n", i3, i2, i1);
						}
					}
				}
			}

			curTwist += fiberTwist * v0.lRest;
			curTwist = glm::mod(curTwist, 2 * glm::pi<float>());
		}

		fclose(file);
	}
}