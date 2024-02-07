
#include "../YarnBall.h"

namespace YarnBall {
	void Sim::step(float h) {
		meta.lastH = meta.h;
		meta.h = h;
		meta.time += h;
		uploadMeta();

		startIterate();

		for (size_t i = 0; i < meta.numItr; i++, frame++) {
			if (meta.collisionPeriod > 0 && frame % meta.collisionPeriod == 0) {
				// TODO: Build collision data
			}

			iterateCosserat();
			// iterateSpring();
		}

		endIterate();
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