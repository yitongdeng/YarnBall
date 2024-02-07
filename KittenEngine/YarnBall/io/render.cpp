#include "../YarnBall.h"
#include <cuda.h>

namespace YarnBall {
	void Sim::render() {
		static auto segBase = Kit::get<Kit::Shader>("resources\\shaders\\yarnSegBase.glsl");
		static auto segForward = Kit::get<Kit::Shader>("resources\\shaders\\yarnSegForward.glsl");

		vertBuffer->bind(5);

		segBase->setInt("numVerts", meta.numVerts);
		segForward->setInt("numVerts", meta.numVerts);

		segBase->setFloat("radius", meta.radius);
		segForward->setFloat("radius", meta.radius);

		Kit::renderInstancedForward(cylMesh, meta.numVerts - 1, segBase, segForward);
	}
}