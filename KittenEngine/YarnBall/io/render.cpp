#include "../YarnBall.h"
#include <cuda.h>

namespace YarnBall {
	void Sim::render() {
		static auto segBase = Kit::get<Kit::Shader>("resources\\shaders\\yarnSegBase.glsl");
		static auto segForward = Kit::get<Kit::Shader>("resources\\shaders\\yarnSegForward.glsl");

		vertBuffer->bind(5);

		segBase->setInt("numVerts", meta.numVerts);
		segForward->setInt("numVerts", meta.numVerts);

		float r = meta.radius + 0.5f * meta.barrierThickness;
		segBase->setFloat("radius", r);
		segForward->setFloat("radius", r);

		Kit::renderInstancedForward(cylMesh, meta.numVerts - 1, segBase, segForward);
	}

	void Sim::renderShadows() {
		static auto segBase = Kit::get<Kit::Shader>("resources\\shaders\\yarnSegBase.glsl");

		vertBuffer->bind(5);
		segBase->setInt("numVerts", meta.numVerts);

		float r = meta.radius + 0.5f * meta.barrierThickness;
		segBase->setFloat("radius", r);

		Kit::renderInstancedShadows(cylMesh, meta.numVerts - 1, segBase);
	}
}