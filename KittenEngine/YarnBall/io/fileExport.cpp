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

		// Export frame (e1, e2, e3)
		std::size_t insertPos = path.size() - 4;
		std::string filename_e1 = path.substr(0, insertPos) + "_e1" + path.substr(insertPos);
		std::string filename_e2 = path.substr(0, insertPos) + "_e2" + path.substr(insertPos);
		std::string filename_e3 = path.substr(0, insertPos) + "_e3" + path.substr(insertPos);
		FILE* pFile_e1 = fopen(filename_e1.c_str(), "w");
		FILE* pFile_e2 = fopen(filename_e2.c_str(), "w");
		FILE* pFile_e3 = fopen(filename_e3.c_str(), "w");
		//printf("Filename %s\n", filename.c_str());

		if (pFile_e1 == NULL or pFile_e2 == NULL or pFile_e3 == NULL) {
			std::cout << "Error opening file" << std::endl;
			return;
		}

		float scale = 0.01;
		for (size_t i = 0; i < meta.numVerts; i++) {
			mat3 rot_mat_i = qs[i].matrix();
			vec3 e1_i = verts[i].pos + scale * vec3(rot_mat_i[0][0], rot_mat_i[0][1], rot_mat_i[0][2]);
			vec3 e2_i = verts[i].pos + scale * vec3(rot_mat_i[1][0], rot_mat_i[1][1], rot_mat_i[1][2]);
			vec3 e3_i = verts[i].pos + scale * vec3(rot_mat_i[2][0], rot_mat_i[2][1], rot_mat_i[2][2]);
			fprintf(pFile_e1, "v %.16f %.16f %.16f\n", verts[i].pos.x, verts[i].pos.y, verts[i].pos.z);
			fprintf(pFile_e1, "v %.16f %.16f %.16f\n", e1_i.x, e1_i.y, e1_i.z);
			fprintf(pFile_e2, "v %.16f %.16f %.16f\n", verts[i].pos.x, verts[i].pos.y, verts[i].pos.z);
			fprintf(pFile_e2, "v %.16f %.16f %.16f\n", e2_i.x, e2_i.y, e2_i.z);
			fprintf(pFile_e3, "v %.16f %.16f %.16f\n", verts[i].pos.x, verts[i].pos.y, verts[i].pos.z);
			fprintf(pFile_e3, "v %.16f %.16f %.16f\n", e3_i.x, e3_i.y, e3_i.z);
		}

		// Parse segments
		lastSeg = false;
		for (size_t i = 0; i < meta.numVerts * 2; i+=2) {
			//printf("i: %i\n", i);
			fprintf(pFile_e1, "l %d %d\n", i + 1, i + 2);
			fprintf(pFile_e2, "l %d %d\n", i + 1, i + 2);
			fprintf(pFile_e3, "l %d %d\n", i + 1, i + 2);
		}

		fclose(pFile_e1);
		fclose(pFile_e2);
		fclose(pFile_e3);
	}
}