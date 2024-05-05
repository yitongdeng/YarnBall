#include "../YarnBall.h"
#include <cuda.h>
#include <fstream>
#include <json/json.h>


namespace YarnBall {
	Sim* buildFromJSON(std::string path) {
		std::ifstream file(path, std::ifstream::binary);
		if (!file.is_open())
			throw std::runtime_error("Error opening file");

		Json::CharReaderBuilder rbuilder;
		std::string errs;

		Json::Value root;
		if (!Json::parseFromStream(rbuilder, file, &root, &errs))
			throw std::runtime_error("Error parsing the JSON file: " + errs);

		Sim* sim = readFromBCC(
			root["curveFile"].asCString(),
			root["resampleLength"].asFloat());

		if (!root["curveRadius"].isNull()) {
			float r = root["curveRadius"].asFloat();
			constexpr float ratio = 0.05f;
			sim->meta.radius = ratio * r;
			sim->meta.barrierThickness = 2 * (1 - ratio) * r;
		}

		auto simRoot = root["simulation"];
		double density = 1.;
		double kStretch = 5e5;
		double kBend = 1e-1;
		if (!simRoot.isNull()) {
			// Material
			if (!simRoot["density"].isNull())
				density = simRoot["density"].asFloat();
			if (!simRoot["frictionCoeff"].isNull())
				sim->meta.frictionCoeff = simRoot["frictionCoeff"].asFloat();
			if (!simRoot["kStretch"].isNull())
				kStretch = simRoot["kStretch"].asFloat();
			if (!simRoot["kBend"].isNull())
				kBend = simRoot["kBend"].asFloat();
			if (!simRoot["kCollision"].isNull())
				sim->meta.kCollision = simRoot["kCollision"].asFloat();
			if (!simRoot["kFriction"].isNull())
				sim->meta.kFriction = simRoot["kFriction"].asFloat();
			if (!simRoot["damping"].isNull())
				sim->meta.damping = simRoot["damping"].asFloat();

			// Environment
			if (!simRoot["gravity"].isNull()) {
				auto g = simRoot["gravity"];
				sim->meta.gravity = glm::vec3(g[0].asFloat(), g[1].asFloat(), g[2].asFloat());
			}
			if (!simRoot["drag"].isNull())
				sim->meta.drag = simRoot["drag"].asFloat();

			// Sim params
			if (!simRoot["maxTimeStep"].isNull())
				sim->maxH = simRoot["maxTimeStep"].asFloat();
			if (!simRoot["numIterations"].isNull())
				sim->meta.numItr = simRoot["numIterations"].asInt();
			if (!simRoot["detectionPeriod"].isNull())
				sim->meta.detectionPeriod = simRoot["detectionPeriod"].asInt();
			if (!simRoot["detectionScaler"].isNull())
				sim->meta.detectionScaler = simRoot["detectionScaler"].asFloat();
			if (!simRoot["stepLimit"].isNull())
				sim->meta.useStepSizeLimit = simRoot["stepLimit"].asBool() ? 1 : 0;
		}

		sim->configure(density);
		sim->setKStretch(kStretch);
		sim->setKBend(kBend);

		if (!root["glueEndpoints"].isNull()) {
			float radius = root["glueEndpoints"].asFloat();
			if (radius >= 0)
				sim->glueEndpoints(radius);
		}

		if (!root["fixBorders"].isNull()) {
			auto borders = root["fixBorders"];
			if (borders.isArray() && borders.size() == 6) {
				float border[6];
				for (int i = 0; i < 6; i++)
					border[i] = borders[i].asFloat();

				// Get bounding box
				Kit::Bound<> bounds;
				for (int i = 0; i < sim->meta.numVerts; i++)
					bounds.absorb(sim->verts[i].pos);

				bounds.pad(0.01f * sim->meta.radius);
				bounds.max.x -= border[0];
				bounds.min.x += border[1];
				bounds.max.y -= border[2];
				bounds.min.y += border[3];
				bounds.max.z -= border[4];
				bounds.min.z += border[5];

				for (int i = 0; i < sim->meta.numVerts; i++)
					if (!bounds.contains(sim->verts[i].pos))
						sim->verts[i].invMass = 0;
			}
		}

		sim->upload();
		return sim;
	}

	void Sim::glueEndpoints(float searchRadius) {
		std::vector<int> endPoints;

		for (int i = 0; i < meta.numVerts; i++) {
			bool hasPrev = i && (verts[i - 1].flags & (int)VertexFlags::hasNext) != 0;
			bool hasNext = (verts[i].flags & (int)VertexFlags::hasNext) != 0;
			if (hasPrev ^ hasNext)
				endPoints.push_back(i);
		}

#pragma omp parallel for schedule(dynamic, 1)
		for (int k = 0; k < endPoints.size(); k++) {
			int i = endPoints[k];
			auto pos = verts[i].pos;

			// Find closest vertex
			float minDist = INFINITY;
			int closest = -1;
			for (int j = 0; j < meta.numVerts; j++)
				if (abs(i - j) > 2) {
					auto dist = glm::length(pos - verts[j].pos);
					if (dist < minDist) {
						minDist = dist;
						closest = j;
					}
				}

			if (minDist <= searchRadius) {
#pragma omp critical 
				{
					if (verts[closest].connectionIndex < 0 && verts[i].connectionIndex < 0) {
						verts[closest].connectionIndex = i;
						verts[i].connectionIndex = closest;
						verts[closest].pos = verts[i].pos = 0.5f * (verts[i].pos + verts[closest].pos);
					}
				}
			}
		}
	}

}