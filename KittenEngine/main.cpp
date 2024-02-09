
#include <cassert>
#include <iostream>

#include "KittenEngine/includes/KittenEngine.h"
#include "KittenEngine/includes/modules/BasicCameraControl.h"

#include "YarnBall/YarnBall.h"

using namespace glm;
using namespace std;

Kit::BasicCameraControl camera;
YarnBall::Sim* sim = nullptr;

bool simulate = false;
float timeScale = 1;
float measuredSimSpeed = 1;

void renderScene() {
	if (simulate) {
		// Dynamic dt
		const float realTime = ImGui::GetIO().DeltaTime * timeScale;
		const float advTime = glm::min(realTime, 1 / 40.f);

		Kit::StopWatch timer;
		sim->advance(advTime);
		float measuredTime = timer.time();

		measuredSimSpeed = mix(measuredSimSpeed, advTime / measuredTime, 0.05f);
	}

	Kit::lights[0].dir = -normalize(Kit::lights[0].pos);
	Kit::projMat = glm::perspective(45.0f, Kit::getAspect(), 0.05f, 512.f);
	Kit::viewMat = camera.getViewMatrix();

	// Render everything
	Kit::startRender();
	glClearColor(0.08f, 0.08f, 0.08f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	sim->render();
}

void renderGui() {
	ImGui::Begin("Control Panel");
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::Text("Measured simulation speed %.2f sim sec per real sec (INCLUDES EXPORT OVERHEAD IF ON)", measuredSimSpeed);

	if (ImGui::TreeNode("Simulation")) {
		ImGui::Checkbox("Simulate", &simulate);

		ImGui::Separator();

		ImGui::SliderFloat("Time Scale", &timeScale, 0.001, 2);
		ImGui::DragFloat3("Gravity", (float*)&sim->meta.gravity);

		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Rendering")) {
		ImGui::DragFloat3("Light Position", (float*)&Kit::lights[0].pos, 0.01f);

		ImGui::TreePop();
	}

	ImGui::End();
}

void initScene() {
	Kit::loadDirectory("resources");

	Kit::UBOLight light;
	light.col = vec4(1, 1, 1, 4);
	light.dir = vec3(sin(radians(30.f)), -cos(radians(30.f)), 0);
	light.pos = -2.f * light.dir;
	light.hasShadow = false;
	light.type = (int)Kit::KittenLight::DIR;
	Kit::lights.push_back(light);

	Kit::ambientLight.col = vec4(0);

	camera.angle = vec2(30, 30);

	if (true) {
		constexpr int numVerts = 64;
		sim = new YarnBall::Sim(numVerts);

		for (size_t i = 0; i < 32; i++)
			sim->verts[i].pos = vec3(0.01f * i, 0, 0);
		// 	sim->verts[i].pos = vec3(0.01f * i, (i % 2) * 0.01f, 0);
		for (size_t i = 0; i < 32; i++)
			sim->verts[i + 32].pos = vec3(0.225f, -0.05, 0.01f * i - 0.16f);

		sim->verts[0].invMass = sim->verts[32].invMass = sim->verts[63].invMass = 0;
		sim->verts[0].flags |= (uint32_t)YarnBall::VertexFlags::fixOrientation;
		sim->verts[31].flags = 0;

		// sim->verts[55].invMass = 0;

		sim->configure();
		sim->setKBend(0.001);
		sim->setKStretch(1e1);
		sim->upload();

		// sim->meta.collisionPeriod = -1;

		sim->meta.radius = 0.002f;
		// sim->maxH = 1 / 30.f;
		camera.pos = sim->verts[0].pos;
	}

	if (!sim) exit(-1);
}

void mouseButtonCallback(GLFWwindow* w, int button, int action, int mode) {
	camera.processMouseButton(button, action, mode);
}

void cursorPosCallback(GLFWwindow* w, double xp, double yp) {
	camera.processMousePos(xp, yp);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	camera.processMouseScroll(xoffset, yoffset);
}

void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mode) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(w, true);

	if (key == GLFW_KEY_F && action == GLFW_PRESS)
		sim->step(sim->maxH);

	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
		simulate = !simulate;
}

int main(int argc, char** argv) {
	// Init window and OpenGL
	Kit::initWindow(ivec2(800, 600), "OpenGL Window");

	// Register callbacks
	Kit::getIO().mouseButtonCallback = mouseButtonCallback;
	Kit::getIO().cursorPosCallback = cursorPosCallback;
	Kit::getIO().scrollCallback = scrollCallback;
	Kit::getIO().keyCallback = keyCallback;

	// Init scene
	initScene();

	while (!Kit::shouldClose()) {
		Kit::startFrame();
		renderScene();		// Render
		renderGui();		// GUI Render
		Kit::endFrame();
	}

	Kit::terminate();
	return 0;
}