
struct Vertex {
	// Linear 
	vec3 pos;				// Node position
	float invMass;			// Inverse nodal mass

	float lRest;			// Rest length
	float stretchK;			// Stretching stiffness
	int connectionIndex;	// Index of connected node -1 if none. (Used to connect vertices)
	int flags;				// Flags see VertexFlags
};

// In in all shader types
layout(binding = 5, std430) buffer vertBlock {
	Vertex verts[];
};

// In in all shader types
layout(binding = 6, std430) buffer qBlock {
	vec4 qs[];
};

