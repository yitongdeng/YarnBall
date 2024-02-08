
#include "../YarnBall.h"

namespace YarnBall {
	void Sim::step(float h) {
		// Set some time step specific parameters
		meta.lastH = meta.h;
		meta.h = h;
		meta.time += h;

		meta.detectionRadius = meta.radius + meta.barrierThickness;
		meta.colGridSize = 0.5f * meta.detectionScaler * length(vec2(meta.maxSegLen + 2 * meta.detectionRadius, 2 * meta.detectionRadius));
		meta.detectionRadius *= meta.detectionScaler;

		if (meta.maxSegLen < 2 * (meta.radius + meta.barrierThickness))
			throw std::runtime_error("Use thinner yarn or use longer segments. (maxSegLen must be at least 2 * (radius + barrierThickness)");

		// Upload parameters and reset error flag
		uploadMeta();
		cudaMemset(d_error, 0, sizeof(int));

		// Solver iterations
		startIterate();

		for (size_t i = 0; i < meta.numItr; i++) {
			if (meta.collisionPeriod > 0 && i % meta.collisionPeriod == 0)
				detectCollisions();

			iterateCosserat();
		}

		endIterate();

		// Error handling
		int error;
		cudaMemcpy(&error, d_error, sizeof(int), cudaMemcpyDeviceToHost);
		if (error == ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED) {
			if (printErrors) fprintf(stderr, "ERROR: MAX_COLLISIONS_PER_SEGMENT exceeded. Current simulation state may be corrupted!\n");
			throw std::runtime_error("MAX_COLLISIONS_PER_SEGMENT exceeded");
		}
		else if (printErrors)
			if (error == WARNING_SEGMENT_STRETCH_EXCEEDS_DETECTION_SCALER)
				fprintf(stderr, "WARNING: Excessive segment stretching detected. Missed collisions possible due to insufficient detection radius.\n");
			else if (error == WARNING_SEGMENT_INTERPENETRATION)
				fprintf(stderr, "WARNING: Some collisions have been temporarily disabled due to interpenetration.\n");
			else if (error != 0)
				fprintf(stderr, "ERROR: Undescript error %d\n", error);

		if (error != 0) lastErrorCode = error;
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