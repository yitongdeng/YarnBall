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
				sim->meta.detectionScaler = simRoot["detectionScaler"].asInt();
		}

		sim->configure(density);
		sim->setKStretch(kStretch);
		sim->setKBend(kBend);
		sim->upload();

		return sim;
	}
}