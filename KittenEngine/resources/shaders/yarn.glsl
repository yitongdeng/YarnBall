
struct Vertex {
	// Linear 
	vec3 pos;				// Node position
	float invMass;			// Inverse nodal mass
	vec3 vel;				// Node velocity
	float lRest;			// Rest length

	// Rotational
	vec4 q;					// Rotation
	vec4 qRest;				// Resting rotation

	float bendK;			// Bending stiffness
	float stretchK;			// Stretching stiffness
	int connectionIndex;	// Index of connected node -1 if none. (Used to connect vertices)
	int flags;				// Flags see VertexFlags
};

// In in all shader types
layout(binding = 5, std430) buffer vertBlock {
	Vertex verts[];
};
