#include "../YarnBall.h"
#include <thrust/reduce.h>
#include "KittenEngine/opt/svd.h"

namespace YarnBall {
	struct LinearMotionSumFunctor {
		__host__ __device__
			LinearMotionSum operator()(const LinearMotionSum& lhs, const LinearMotionSum& rhs) const {
			LinearMotionSum sum;
			for (int i = 0; i < 3; i++)
				sum.rhs[i] = lhs.rhs[i] + rhs.rhs[i];
			sum.outerSum = lhs.outerSum + rhs.outerSum;
			return sum;
		}
	};

	void Sim::transferMotion() {
		// Do reduction
		LinearMotionSum sum;
		sum.outerSum = Kit::hess4(0);
		sum.rhs[0] = sum.rhs[1] = sum.rhs[2] = vec4(0);
		auto d_ptr = thrust::device_ptr<LinearMotionSum>(meta.d_motions);
		sum = thrust::reduce(d_ptr, d_ptr + meta.numVerts, sum, LinearMotionSumFunctor());

		mat4 invMat = inverse((mat4)sum.outerSum);
		mat3 m;
		vec3 c;
		for (int i = 0; i < 3; i++) {
			vec4 r = invMat * sum.rhs[i];
			m[i] = vec3(r);
			c[i] = r.w;
		}

		m = transpose(m);

		// Should clamp eigen values to be greater than -1
		// In theory this should be very rare.
		mat3 U, V;
		vec3 S;
		Kit::svd(m, U, S, V);
		for (int i = 0; i < 3; i++) {
			float s = sign(dot(U[i], V[i]));
			S[i] *= s;
			V[i] *= s;
			S[i] = max(S[i], -0.999f);
		}
		m = Kit::svdMul(U, S, V);

		meta.linearMotionMatrix = m;
		meta.linearMotionVector = c;
	}

	__global__ void testMotionDeformKernel(MetaData* data, mat3 m, vec3 c) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		vec3 pos = data->d_verts[tid].pos;
		data->d_verts[tid].pos = pos + (m * pos + c) * 5.0f;
	}

	void Sim::testMotionDeform() {
		testMotionDeformKernel << <(meta.numVerts + 255) / 256, 256 >> > (d_meta, meta.linearMotionMatrix, meta.linearMotionVector);
		cudaDeviceSynchronize();
	}
}