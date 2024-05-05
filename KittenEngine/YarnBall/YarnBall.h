#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>

#include "KittenEngine/includes/KittenEngine.h"
#include "KittenEngine/KittenGpuLBVH/lbvh.cuh"
#include "KittenEngine/includes/modules/Bound.h"

namespace YarnBall {
	using namespace glm;

	// This should really NEVER be exceeded.
	constexpr int MAX_COLLISIONS_PER_SEGMENT = 128;

	// BCC file header
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

	enum class VertexFlags {
		hasPrev = 1,			// Whether the vertex has a previous vertex
		hasNext = 2,			// Whether the vertex has a next vertex
		hasNextOrientation = 4,	// Whether the segment has a next segment
		fixOrientation = 8,		// Fix the orientation of the segment
		colliding = 16,			// Whether this is colliding (unused)
	};

	inline bool hasFlag(const uint32_t flags, const VertexFlags flag) {
		return (flags & (uint32_t)flag) != 0;
	}

	inline uint32_t setFlag(const uint32_t flags, const VertexFlags flag, const bool state) {
		return state ? flags | (uint32_t)flag : flags & ~(uint32_t)flag;
	}

	// Simulation vertex (aligned to openGL layout)
	// This includes everything needed to form the local hessian minus collisions
	typedef struct {
		// Linear 
		vec3 pos;			// Node position
		float invMass;		// Inverse nodal mass
		vec3 vel;			// Node velocity
		float lRest;		// Rest length

		// Rotational
		Kit::Rotor q;		// Rotation
		vec4 qRest;			// Resting rotation

		float kBend;		// Bending stiffness
		float kStretch;		// Stretching stiffness
		int connectionIndex;// Index of connected node -1 if none. (Used to connect vertices)
		uint32_t flags;		// Flags see VertexFlags
	} Vertex;

	typedef struct {
		Vertex* d_verts;		// Device vertex array pointer
		vec3* d_dx;				// Temporary delta position iterants. Stored as deltas for precision.
		vec3* d_lastVels;		// Velocity from the last frame

		vec3* d_lastPos;		// Last vertex positions. Temp storage to speed up memory access.
		uint32_t* d_lastFlags;	// Last vertex flags. Temp storage to speed up memory access.
		int* d_lastCID;			// Last cid. Temp storage to speed up memory access.

		int* d_numCols;					// Number of collisions for each segment
		float* d_maxStepSize;			// Max step size for each vertex
		int* d_collisions;				// Collisions IDs stored as the other segment index.
		Kit::LBVH::aabb* d_bounds;		// AABBs
		ivec2* d_boundColList;			// Colliding segment AABB IDs. 

		vec3 gravity;			// Gravity
		int numItr;				// Number of iterations used per time step

		vec3 worldFloorNormal;	// World floor normal
		float worldFloorPos;	// World floor position

		float h;				// Time step (automatically set)
		float lastH;			// Last time step
		float time;				// Current time
		int numVerts;			// Number of vertices

		float damping;			// Damping forces
		float drag;				// Velocity decay
		float frictionCoeff;	// Friction coefficient for contacts
		float kCollision;		// Stiffness of the collision
		float kFriction;		// Stiffness of the collision

		float detectionRadius;			// Total detection radius of the yarn (automatically set)
		float scaledDetectionRadius;	// Detection radius scaled by the detectionScaler
		float radius;					// Yarn radius. Note that this is the minimum radius. The actual radius is r + 0.5 * barrierThickness
		float accelerationRatio;		// Solver acceleration ratio

		float barrierThickness;	// Collision energy barrier thickness. This is the barrier between yarns.
		float detectionScaler;	// The extra room needed for a close by potential collision to be added as a ratio
		float bvhRebuildPeriod;	// The time in between rebuilding the BVH.
		int detectionPeriod;	// The number of steps in between to perform collision detection. -1 to turn off collisions

		float maxSegLen;		// Largest segment length
		float minSegLen;		// Largest segment length
		int useStepSizeLimit;	// Whether to use the step size limit
	} MetaData;

	class Sim {
	public:
		enum {
			ERROR_NONE = 0,
			ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED,
			WARNING_SEGMENT_INTERPENETRATION
		};

		Vertex* verts;
		MetaData meta;
		float maxH = 1e-3;						// Largest time step allowed
		std::vector<float> initialInvMasses;	// Starting inverse masses saved for pinned vertices.
		Kit::Bound<> currentBounds;				// Current bounding box

		int lastErrorCode = ERROR_NONE;
		int lastWarningCode = ERROR_NONE;

		bool printErrors = true;
		bool renderShaded = false;

	private:
		MetaData* d_meta = nullptr;
		int* d_error = nullptr;
		bool initialized = false;
		Kit::LBVH bvh;

		float lastBVHRebuild = std::numeric_limits<float>::infinity();
		int lastItr = -1;
		size_t stepCounter = 0;

		// GL stuff
		Kitten::Mesh* cylMesh = nullptr;
		Kitten::Mesh* cylMeshHiRes = nullptr;
		Kitten::CudaComputeBuffer* vertBuffer = nullptr;

		cudaStream_t stream = nullptr;
		cudaGraphExec_t stepGraph = nullptr;

	public:
		Sim(int numVerts);
		~Sim();

		// Initializes memory and sets up rest length, angles, and mass
		void configure(float density = 1e-3);
		void setKBend(float k = 3e-9);
		void setKStretch(float k = 1e-2);

		// Rendering
		void render();
		void renderShadows();

		// Utils
		// void glueEndpoints(float searchRadius);
		void upload();
		void download();
		void zeroVelocities();

		// Simulation
		void step(float dt);
		float advance(float dt);

		void printCollisionStats();
		Kitten::LBVH::aabb bounds();

		void exportToBCC(std::string path);
		void exportToOBJ(std::string path);

		// Glue endpoints with a vertex within the search radius
		void glueEndpoints(float searchRadius);

	private:
		void uploadMeta();

		void startIterate();
		void endIterate();
		void detectCollisions();
		void iterateCosserat();
		void iterateSpring();
		void transferSegmentData();
		void recomputeStepLimit();
		void checkErrors();

		void rebuildCUDAGraph();
	};

	Sim* readFromBCC(std::string path, float targetSegLen);
	Sim* buildFromJSON(std::string path);
}