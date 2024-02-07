#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>

#include "KittenEngine/includes/KittenEngine.h"

namespace YarnBall {
	using namespace glm;

	enum class VertexFlags {
		hasPrev = 1,		// Whether the vertex has a previous vertex
		hasNext = 2,		// Whether the vertex has a next vertex
		fixOrientation = 4,	// Fix the orientation of the segment
	};

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

		float bendK;		// Bending stiffness
		float stretchK;		// Stretching stiffness
		int connectionIndex;// Index of connected node -1 if none. (Used to connect vertices)
		uint32_t flags;		// Flags see VertexFlags
	} Vertex;

	typedef struct {
		Vertex* d_verts;		// Device vertex array pointer
		vec3* d_dx;				// Temporary delta position iterants. Stored as delta for precision.
		vec3* d_lastVels;		// Velocity from the last frame

		vec3 gravity;			// Gravity
		int numItr;				// Number of iterations used per time step

		vec3 worldFloorNormal;	// World floor normal
		float worldFloorPos;	// World floor position

		float h;				// Time step (automatically set)
		float lastH;			// Last time step
		float time;				// Current time

		float drag;				// Velocity decay
		float damping;			// Damping forces
		float frictionCoeff;	// Friction coefficient for contacts

		int collisionPeriod;	// The number of frames in between to check for collisions. -1 to turn off collisions
		float numVerts;			// Number of vertices
		float radius;			// Yarn radius

		float minRadius;		// The minimum yarn distance (radius) maintained in collisions
		float maxSegLen;		// Largest segment length
		float colGridSize;		// Collision grid size
	} MetaData;

	class Sim {
	public:
		Vertex* verts;

		MetaData meta;

		float maxH = 1e-3;		// Largest time step allowed

	private:
		MetaData* d_meta = nullptr;
		bool initialized = false;

		// GL stuff
		Kitten::Mesh* cylMesh = nullptr;
		Kitten::CudaComputeBuffer* vertBuffer = nullptr;

	public:
		Sim(int numVerts);
		~Sim();

		// Initializes memory and sets up rest length, angles, and mass
		void configure(float density = 1);

		void setKBend(float k = 0.5f);
		void setKStretch(float k = 1e2f);

		// IO
		// void readBCC(std::string path);
		// void writeBCC(std::string path);

		// Rendering
		void render();
		// void renderShadow();

		// Utils
		// void glueEndpoints(float searchRadius);
		void upload();
		void download();
		void zeroVelocities();

		// Simulation
		void step(float dt);
		float advance(float dt);

	private:
		void uploadMeta();

		void startIterate();
		void endIterate();
		void iterateCosserat();
		void iterateSpring();
	};

	// Sim* buildFromJSON(std::string path);
}