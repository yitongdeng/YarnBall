#include "../YarnBall.h"
#include <vector>
#include <iostream>

namespace YarnBall {
	void Sim::exportToBCC(std::string path, bool exportAsPolyline) {
		download();
		FILE* pFile;
		pFile = fopen(path.c_str(), "wb");

		if (pFile == NULL) {
			std::cout << "Error opening file" << std::endl;
			return;
		}

		BCCHeader header;

		header.sign[0] = 'B';
		header.sign[1] = 'C';
		header.sign[2] = 'C';
		header.byteCount = 0x44;
		if (exportAsPolyline) {
			header.curveType[0] = 'P';
			header.curveType[1] = 'L';
		}
		else {
			header.curveType[0] = 'C';
			header.curveType[1] = '0';
		}
		header.dimensions = 3;
		header.upDimension = 1;

		// Parse segments
		bool lastSeg = false;
		std::vector<ivec2> segs;
		size_t start = 0;
		for (size_t i = 0; i < meta.numVerts; i++) {
			bool seg = (verts[i].flags & (uint32_t)VertexFlags::hasNext) != 0;
			if (seg) {
				if (!lastSeg)
					start = i;
			}
			else if (lastSeg)
				segs.push_back(ivec2(start, i));
			lastSeg = seg;
		}

		// Count segments
		header.curveCount = segs.size();
		header.totalControlPointCount = 0;
		for (auto seg : segs)
			header.totalControlPointCount += seg.y - seg.x + 1;

		// Write segments
		fwrite(&header, sizeof(BCCHeader), 1, pFile);
		for (auto seg : segs) {
			int numPoints = seg.y - seg.x + 1;
			fwrite(&numPoints, sizeof(int), 1, pFile);
			for (size_t i = seg.x; i <= seg.y; i++)
				fwrite(&verts[i].pos, sizeof(vec3), 1, pFile);
		}

		fclose(pFile);
	}

	void Sim::exportToOBJ(std::string path) {
		download();
		FILE* pFile;
		pFile = fopen(path.c_str(), "w");

		if (pFile == NULL) {
			std::cout << "Error opening file" << std::endl;
			return;
		}

		fprintf(pFile, "# YarnBall Sim\n\n");
		fprintf(pFile, "o YarnBall\n\n");
		fprintf(pFile, "# Vertices (in meters)\n");
		for (size_t i = 0; i < meta.numVerts; i++)
			fprintf(pFile, "v %.16f %.16f %.16f\n", verts[i].pos.x, verts[i].pos.y, verts[i].pos.z);
		fprintf(pFile, "\n# Curves\n");

		// Parse segments
		bool lastSeg = false;
		for (size_t i = 0; i < meta.numVerts; i++) {
			bool seg = (verts[i].flags & (uint32_t)VertexFlags::hasNext) != 0;
			if (seg) {
				if (!lastSeg) {
					fprintf(pFile, "l %d", i + 1);
				}
				fprintf(pFile, " %d", i + 2);
			}
			else if (lastSeg) {
				fprintf(pFile, "\n");
			}
			lastSeg = seg;
		}

		fclose(pFile);
	}
}