
#include "../YarnBall.h"

namespace YarnBall {
	void Sim::rebuildCUDAGraph() {
		// Graph is still good
		if (meta.collisionPeriod == lastColPeriod && meta.numItr == lastItr)
			return;
		checkCudaErrors(cudaGetLastError());

		// Build graph
		cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

		// Solver iterations
		startIterate();

		for (size_t i = 0; i < meta.numItr; i++) {
			if (meta.collisionPeriod > 0 && i % meta.collisionPeriod == 0)
				detectCollisions();
			iterateCosserat();
		}

		endIterate();

		cudaGraph_t graph;
		cudaStreamEndCapture(stream, &graph);

		// Create graph data
		if (stepGraph) cudaGraphExecDestroy(stepGraph);
		cudaGraphInstantiate(&stepGraph, graph, NULL, NULL, 0);
		cudaGraphDestroy(graph);

		// Force reset collisions if we explicitly disabled them
		if (meta.collisionPeriod <= 0)
			cudaMemsetAsync(meta.d_numCols, 0, sizeof(int) * meta.numVerts, stream);

		checkCudaErrors(cudaGetLastError());

		lastColPeriod = meta.collisionPeriod;
		lastItr = meta.numItr;
	}

	float Sim::advance(float h) {
		if (h <= 0) return 0;

		int steps = max(1, (int)ceil(h / maxH));
		meta.lastH = meta.h;
		meta.h = h / steps;

		rebuildCUDAGraph();
		uploadMeta();

		for (int s = 0; s < steps; s++)
			cudaGraphLaunch(stepGraph, stream);

		meta.time += h;
		checkErrors();
	}

	void Sim::step(float h) {
		advance(maxH);
	}
}