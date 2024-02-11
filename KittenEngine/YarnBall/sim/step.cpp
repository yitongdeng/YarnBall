
#include "../YarnBall.h"

namespace YarnBall {
	void Sim::step(float h) {
		// Set some time step specific parameters
		meta.lastH = meta.h;
		meta.h = h;
		meta.time += h;

		meta.detectionRadius = meta.radius + 0.5f * meta.barrierThickness;
		meta.colGridSize = 0.5f * meta.detectionScaler * length(vec2(meta.maxSegLen + 2 * meta.detectionRadius, 2 * meta.detectionRadius));
		meta.detectionRadius *= meta.detectionScaler;

		if (meta.maxSegLen < 2 * (meta.radius + meta.barrierThickness))
			throw std::runtime_error("Use thinner yarn or use longer segments. (maxSegLen must be at least 2 * (radius + barrierThickness)");

		// Upload parameters
		uploadMeta();

		// Force reset collisions if we explicitly disabled them
		if (meta.collisionPeriod <= 0)
			cudaMemset(meta.d_numCols, 0, sizeof(int) * meta.numVerts);

		// Solver iterations
		startIterate();

		for (size_t i = 0; i < meta.numItr; i++) {
			if (meta.collisionPeriod > 0 && i % meta.collisionPeriod == 0)
				detectCollisions();
			iterateCosserat();
		}

		endIterate();
		checkErrors();
	}

	float Sim::advance(float h) {
		if (h <= 0) return 0;
		float hLeft = h;
		int stepsLeft;

		while (true) {
			stepsLeft = (int)ceil(hLeft / maxH);
			float stepSize = hLeft / stepsLeft;

			step(stepSize);
			hLeft -= stepSize;
			stepsLeft--;
			if (stepsLeft <= 0)
				break;
		}

		return h - hLeft;
	}
}