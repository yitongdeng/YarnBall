#include "../YarnBall.h"
#include <cuda.h>
#include <vector>

using namespace std;

namespace YarnBall {
	struct BCCHeader {
		char sign[3];
		unsigned char byteCount;
		char curveType[2];
		char dimensions;
		char upDimension;
		uint64_t curveCount;
		uint64_t totalControlPointCount;
		char fileInfo[40];
	};

	// Simple algorithm to resample a CMR spline
	vector<vec3> resampleCMR(vector<vec3> cmr, float tarSegLen) {
		vector<vec3> resampled;

		int i = 1;
		float lenLeft = 0;
		while (i < cmr.size() - 2) {
			vec3 p0 = cmr[i - 1];
			vec3 p1 = cmr[i];
			vec3 p2 = cmr[i + 1];
			vec3 p3 = cmr[i + 2];

			// Estimate the arc length of the curve segment
			constexpr int numSamples = 3;
			vec3 lastPos = p1;
			float arcLen = 0;
			for (size_t j = 0; j < numSamples; j++) {
				vec3 pos = Kit::cmrSpline(p0, p1, p2, p3, (j + 1) / (float)(numSamples + 1));
				arcLen += glm::length(pos - lastPos);
				lastPos = pos;
			}
			arcLen += glm::length(p2 - lastPos);

			float t = lenLeft / arcLen;
			float step = tarSegLen / arcLen;

			// Set t limit. This is so we leave enough room for the last point
			float maxT = (i < cmr.size() - 3) ? 1 : 1 - step;
			while (t < maxT) {
				vec3 pos = Kit::cmrSpline(p0, p1, p2, p3, t);
				resampled.push_back(pos);
				t += step;
			}
			lenLeft = (t - 1) * arcLen;
			i++;
		}

		// Add the last point on there
		resampled.push_back(cmr[cmr.size() - 2]);
		return resampled;
	}

	Sim* readFromBCC(std::string path, float targetSegLen) {
		BCCHeader header;
		FILE* pFile = fopen(path.c_str(), "rb");
		if (!pFile) throw std::runtime_error("Could not open file");
		fread(&header, sizeof(header), 1, pFile);

		// Error checking
		if (header.sign[0] != 'B' || header.sign[1] != 'C' || header.sign[2] != 'C' || header.byteCount != 0x44) {
			fprintf(stderr, "Unsupported BCC file\n");
			throw std::runtime_error("Unsupported BCC file");
		}

		bool isPolyline = header.curveType[0] == 'P' && header.curveType[1] == 'L';
		if (!isPolyline && (header.curveType[0] != 'C' || header.curveType[1] != '0'))
			throw std::runtime_error("Not polyline or uniform CMR spline");
		if (header.dimensions != 3)
			throw std::runtime_error("Only curves in 3D are supported");

		vector<vector<vec3>> curves;
		vector<bool> isCurveClosed;
		int numVerts = 0;

		// Read file into memory
		for (size_t i = 0; i < header.curveCount; i++) {
			int numPoints;
			fread(&numPoints, sizeof(int), 1, pFile);
			bool isClosed = numPoints < 0;
			numPoints = abs(numPoints);

			vector<vec3> points(numPoints);
			fread(&points[0], sizeof(vec3), numPoints, pFile);

			// Convert to meters from cm
			for (auto& p : points) p *= 0.01f;

			if (!isPolyline)	// Resample CMR spline
				points = resampleCMR(points, targetSegLen);

			// Ignore curves with less than 3 points
			if (numPoints < 3) continue;
			if (isClosed) points.push_back(points[0]);

			isCurveClosed.push_back(isClosed);
			curves.push_back(points);
			numVerts += points.size();
		}

		fclose(pFile);

		// Convert to simulation format
		Sim* sim = new Sim(numVerts);
		numVerts = 0;
		float maxLen = 0;
		float minLen = FLT_MAX;
		for (size_t i = 0; i < curves.size(); i++) {
			auto& data = curves[i];
			Vertex* verts = sim->verts + numVerts;

			// Add the vertices
			vec3 lastP = data[0];
			verts[0].pos = lastP;
			for (size_t j = 1; j < data.size(); j++) {
				vec3 p = data[j];
				verts[j].pos = p;
				float l = glm::length(p - lastP);
				maxLen = glm::max(maxLen, l);
				minLen = glm::min(minLen, l);
				lastP = p;
			}

			// Cut off the end
			verts[data.size() - 1].flags = 0;

			// Connect the ends of closed curves
			if (isCurveClosed[i]) {
				verts[0].connectionIndex = numVerts + data.size() - 1;
				verts[data.size() - 1].connectionIndex = numVerts;
			}

			numVerts += data.size();
		}

		printf("Resampled with target length %f. (max %f, min %f)\n", targetSegLen, maxLen, minLen);

		if (maxLen > 1.2f * targetSegLen)
			printf("WARNING: Resampled max len is significantly larger than target length.");
		if (1.2f * minLen < targetSegLen)
			printf("WARNING: Resampled min len is significantly smaller than target length.");

		return sim;
	}
}