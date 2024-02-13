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

	dvec3 sampleCurve(vector<vec3>& cmr, double t) {
		int si = glm::clamp((int)floor(t), 1, (int)cmr.size() - 3);
		dvec3 p0 = cmr[si - 1];
		dvec3 p1 = cmr[si];
		dvec3 p2 = cmr[si + 1];
		dvec3 p3 = cmr[si + 2];

		return Kit::cmrSpline(p0, p1, p2, p3, t - si);
	}

	// Simple algorithm to resample a CMR spline
	vector<vec3> resampleCMR(vector<vec3>& cmr, float tarSegLen) {
		vector<double> ts;

		int i = 1;
		dvec3 lastPos = cmr[i];				// Last position added to the list
		ts.push_back(1);

		while (i < cmr.size() - 2) {
			dvec3 p0 = cmr[i - 1];
			dvec3 p1 = cmr[i];
			dvec3 p2 = cmr[i + 1];
			dvec3 p3 = cmr[i + 2];

			// Repeated find a monotonically increasing t 
			// s.t the distance from the last point is always exactly tarSegLen

			constexpr int numSamples = 3;	// Min samples to take
			dvec3 s0 = p1;		// Last position where the distance value was calculated
			double t0 = 0;

			for (int j = 0; j < numSamples; j++) {
				const double t1 = (j + 1) / (double)numSamples;
				const dvec3 s1 = Kit::cmrSpline(p0, p1, p2, p3, t1);
				double l1 = length(s1 - lastPos);

				// The next sample is outside the sphere
				double l0 = length(s0 - lastPos);
				while (l1 >= tarSegLen) {
					double t = t1 + (t1 - t0) * (tarSegLen - l1) / (l1 - l0);

					for (int k = 0; k < 16; k++) {
						dvec3 x = Kit::cmrSpline(p0, p1, p2, p3, t);
						double d = length(x - lastPos);
						t += (t - t0) * (tarSegLen - d) / (d - l0);
					}

					ts.push_back(i + t);
					lastPos = Kit::cmrSpline(p0, p1, p2, p3, t);
					l1 = length(s1 - lastPos);

					l0 = 0;
					t0 = t;
				}

				t0 = t1;
				s0 = s1;
			}

			i++;
		}

		if (true) {
			// Check how far the last point is from the end
			// Add a new point if its easier to do so
			double d = length(sampleCurve(cmr, ts.back()) - (dvec3)cmr[cmr.size() - 2]);
			double dt = ts[ts.size() - 1] - ts[ts.size() - 2];
			if (d > 0.5f * tarSegLen)
				ts.push_back(cmr.size() - 2);
			else
				ts.back() = cmr.size() - 2;
			dt -= ts[ts.size() - 1] - ts[ts.size() - 2];

			// Now we need to go back and shift somt points to make the last point line up
			const int N = std::min((int)ts.size(), 16);
			for (int i = 0; i < N; i++)
				ts[ts.size() - 2 - i] -= dt * (N - i) / (double)N;
		}

		vector<vec3> resampled(ts.size());
		for (size_t i = 0; i < ts.size(); i++)
			resampled[i] = (vec3)sampleCurve(cmr, ts[i]);
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