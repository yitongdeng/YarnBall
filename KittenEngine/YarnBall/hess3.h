#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "KittenEngine/includes/KittenEngine.h"

namespace YarnBall {
	using namespace glm;

	// Simply 3x3 hessian.
	// There is a templated version of this but the compiler doesnt unroll it properly
	// So here is a simple version where whats needed is hand unrolled.
	struct hess3 {
		union {
			float dat[6];
			struct {
				vec3 diag;
				vec3 upperTriangle;
			};
		};

		KITTEN_FUNC_DECL hess3() {}
		KITTEN_FUNC_DECL hess3(float d) : dat{ d, d, d } {}
		KITTEN_FUNC_DECL hess3(const hess3& m) {
			for (int i = 0; i < 6; i++)
				dat[i] = m[i];
		}

		KITTEN_FUNC_DECL hess3& operator=(const hess3& m) {
			for (int i = 0; i < 6; i++)
				dat[i] = m[i];
			return *this;
		}

		KITTEN_FUNC_DECL float& operator[](const int i) {
			return dat[i];
		}

		KITTEN_FUNC_DECL const float& operator[](const int i) const {
			return dat[i];
		}

		KITTEN_FUNC_DECL hess3 operator+(const hess3& m) const {
			hess3 r;
			for (int i = 0; i < 6; i++)
				r[i] = dat[i] + m[i];
			return r;
		}

		KITTEN_FUNC_DECL hess3 operator-(const hess3& m) const {
			hess3 r;
			for (int i = 0; i < 6; i++)
				r[i] = dat[i] - m[i];
			return r;
		}

		KITTEN_FUNC_DECL hess3& operator+=(const hess3& m) {
			for (int i = 0; i < 6; i++)
				dat[i] += m[i];
			return *this;
		}

		KITTEN_FUNC_DECL hess3& operator-=(const hess3& m) {
			for (int i = 0; i < 6; i++)
				dat[i] -= m[i];
			return *this;
		}

		KITTEN_FUNC_DECL hess3 operator*(const float s) const {
			hess3 r;
			for (int i = 0; i < 6; i++)
				r[i] = dat[i] * s;
			return r;
		}

		KITTEN_FUNC_DECL hess3 operator/(const float s) const {
			float invS = 1.0f / s;
			hess3 r;
			for (int i = 0; i < 6; i++)
				r[i] = dat[i] * invS;
			return r;
		}

		KITTEN_FUNC_DECL hess3 operator-() const {
			hess3 r;
			for (int i = 0; i < 6; i++)
				r[i] = -dat[i];
			return r;
		}

		KITTEN_FUNC_DECL static hess3 outer(vec3 v) {
			hess3 m;
			m[0] = v[0] * v[0];
			m[1] = v[1] * v[1];
			m[2] = v[2] * v[2];
			m[3] = v[0] * v[1];
			m[4] = v[0] * v[2];
			m[5] = v[1] * v[2];
			return m;
		}

		KITTEN_FUNC_DECL explicit operator mat3() const {
			mat3 m;
			m[0][0] = dat[0];
			m[1][1] = dat[1];
			m[2][2] = dat[2];
			m[0][1] = m[1][0] = dat[3];
			m[0][2] = m[2][0] = dat[4];
			m[1][2] = m[2][1] = dat[5];
			return m;
		}
	};
}