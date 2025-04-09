
#include <cassert>
#include <iostream>
#include <CLI/CLI.hpp>

#include "KittenEngine/includes/KittenEngine.h"
#include "KittenEngine/includes/modules/BasicCameraControl.h"
#include "KittenEngine/includes/modules/Dist.h"

#include "YarnBall/YarnBall.h"

using namespace glm;
using namespace std;

Kit::BasicCameraControl camera;
YarnBall::Sim* sim = nullptr;

bool simulate = false;
float timeScale = 1.;
float measuredSimSpeed = 1;
Kit::Dist simSpeedDist;

vector<vec3> initialPos;
vector<Kit::Rotor> initialQ;
Kit::Bound<> initialBounds;

bool exitWhenDone = false;
bool exportSim = false;
bool scenarioTwist = false;
bool scenarioPull = false;
bool exportFiberLevel = false;
bool exportBCC = false;
bool exportEndFrame = false;
float EXPORT_DT = 1 / 30.f;
int exportLimit = 2000;
bool headlessMode = false;
string exportPath = "./frames/frame_";

vec3 rotateY(vec3 v, float angle) {
	return vec3(cos(angle) * v.x - sin(angle) * v.z, v.y, sin(angle) * v.x + cos(angle) * v.z);
}

void performSim() {
	// Dynamic dt
	float advTime = EXPORT_DT;
	if (!headlessMode) {
		const float realTime = ImGui::GetIO().DeltaTime * timeScale;
		if (!exportSim)
			advTime = glm::min(realTime, 1 / 40.f);
	}
	vec3 center = 0.5f * (initialBounds.max + initialBounds.min);

	// Twisting animation
	if (scenarioTwist) {
		sim->download();

		static float twistTime = 0;
		float nextTime = twistTime + advTime;

		constexpr float speed = 0.5f * 2 * glm::pi<float>();
		constexpr float end = 12.f;
		float angle = glm::clamp(twistTime - 2.f, 0.f, end) * speed;
		float nextAngle = glm::clamp(nextTime - 2.f, 0.f, end) * speed;

		for (size_t i = 0; i < sim->meta.numVerts; i++) {
			auto& vert = sim->verts[i];
			auto init = initialPos[i];
			if (vert.pos.y < center.y && vert.invMass == 0) {
				vec3 pos = center + rotateY(init - center, angle);
				vec3 nextPos = center + rotateY(init - center, nextAngle);

				vert.pos = pos;
				sim->vels[i] = (nextPos - pos) / advTime;

				if (twistTime > end + 3.0f) vert.invMass = sim->initialInvMasses[i];
			}
		}
		/*
		if (twistTime > end + 13.f) {
			exportSim = false;
			scenarioTwist = false;
			simulate = false;
		}
		*/
		sim->upload();
		twistTime = nextTime;
	}

	// Pulling animation
	if (scenarioPull) {
		sim->download();
		sim->meta.gravity = vec3(0, 0, -9.8);
		static float pullTime = 0;
		float nextTime = pullTime + advTime;

		const float speed = 0.25f;
		const float end = 6.f;
		float x = -speed * glm::clamp(pullTime - 2.f, 0.f, end);
		float nextX = -speed * glm::clamp(nextTime - 2.f, 0.f, end);

		for (size_t i = 0; i < sim->meta.numVerts; i++) {
			auto& vert = sim->verts[i];
			auto init = initialPos[i];
			if (vert.invMass == 0) {
				float pos = init.y + ((vert.pos.y < center.y) ? x : -x);
				float nextPos = init.y + ((vert.pos.y < center.y) ? nextX : -nextX);

				vert.pos.y = pos;
				sim->vels[i].y = (nextPos - pos) / advTime;
			}
		}
		/*
		if (pullTime > end + 4.f) {
			exportSim = false;
			scenarioPull = false;
			simulate = false;
		}
		*/
		sim->upload();
		pullTime = nextTime;
	}

	Kit::StopWatch timer;
	sim->advance(advTime);
	float measuredTime = timer.time();

	if (exportSim) {
		static int frameID = 0;
		if (frameID > exportLimit) {
			exportSim = false;
			simulate = false;
			if (exportEndFrame)
				if (exportFiberLevel) sim->exportFiberMesh(exportPath);
				else if (exportBCC) sim->exportToBCC(exportPath, false);
				else sim->exportToOBJ(exportPath);

			printf("Export complete. sim/real ratio Avg %.3f, SD: %.3f, N=%d\n", simSpeedDist.mean(), simSpeedDist.sd(), simSpeedDist.num);
			if (exitWhenDone) exit(0);
		}
		if (!exportEndFrame)
			if (exportFiberLevel) sim->exportFiberMesh(exportPath + to_string(frameID) + ".obj");
			else if (exportBCC) sim->exportToBCC(exportPath + to_string(frameID) + ".bcc", false);
			else sim->exportToOBJ(exportPath + to_string(frameID) + ".obj");
		frameID++;
	}

	float ss = advTime / measuredTime;
	measuredSimSpeed = mix(measuredSimSpeed, ss, 0.1f);
	simSpeedDist.accu(ss);
}

void renderScene() {
	auto bounds = sim->bounds();
	if (glm::isfinite(bounds.min.x)) {
		Kit::lights[0].pos = bounds.center();
		Kit::shadowDist = length(bounds.max - bounds.min) * 0.5f;
	}
	Kit::projMat = glm::perspective(45.0f, Kit::getAspect(), 0.005f, 512.f);
	Kit::viewMat = camera.getViewMatrix();

	// Render everything
	Kit::startRender();
	glClearColor(0.08f, 0.08f, 0.08f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	sim->startRender();
	sim->renderShadows();
	sim->render();
}

void renderGui() {
	ImGui::Begin("Control Panel");
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::Text("Measured simulation speed %.3f sim sec per real sec (INCLUDES EXPORT OVERHEAD IF ON)", measuredSimSpeed);
	ImGui::Text("Avg %.3f, SD: %.3f, N=%d", simSpeedDist.mean(), simSpeedDist.sd(), simSpeedDist.num);
	ImGui::Text("Sim time: %.3f", sim->meta.time);

	if (ImGui::TreeNode("Simulation")) {
		ImGui::Checkbox("Simulate", &simulate);

		ImGui::Separator();

		ImGui::SliderFloat("Time Scale", &timeScale, 0.001, 2);
		ImGui::DragFloat3("Gravity", (float*)&sim->meta.gravity);
		ImGui::Separator();

		ImGui::InputFloat("Max dt", &sim->maxH, 1e-4, 1e-3, "%.5f");
		ImGui::InputInt("Itr", &sim->meta.numItr);

		ImGui::Separator();
		if (sim->meta.detectionPeriod >= 0 && ImGui::Button("Disable collisions"))
			sim->meta.detectionPeriod = -1;
		if (ImGui::Button("Zero velocity"))
			sim->zeroVelocities();
		ImGui::Separator();
		ImGui::TreePop();
	}

	static bool lightFollowCam = true;
	static vec3 lightDir = normalize(vec3(1, -2, -1));

	if (ImGui::TreeNode("Rendering")) {
		ImGui::Checkbox("Render shaded", &sim->renderShaded);
		ImGui::Checkbox("Render shadows", (bool*)&Kit::lights[0].hasShadow);

		ImGui::Checkbox("Light follows camera", &lightFollowCam);
		ImGui::DragFloat3("Light direction", (float*)&lightDir);

		ImGui::TreePop();
	}
	if (lightFollowCam)
		Kit::lights[0].dir = (mat3)glm::rotate(mat4(1), radians(camera.angle.x), vec3(0, 1, 0)) * lightDir;
	else
		Kit::lights[0].dir = lightDir;

	if (ImGui::TreeNode("Export")) {
		ImGui::Checkbox("Export", &exportSim);
		ImGui::Checkbox("Export as Fiber Mesh", &exportFiberLevel);
		ImGui::Checkbox("Twist", &scenarioTwist);
		ImGui::Checkbox("Pull", &scenarioPull);
		ImGui::Separator();
		if (ImGui::Button("Export fiber mesh now"))
			sim->exportFiberMesh("./frameFiber.obj");
		if (ImGui::Button("Export frame obj now"))
			sim->exportToOBJ("./frame.obj");
		if (ImGui::Button("Export frame bcc now"))
			sim->exportToBCC("./frame.bcc", false);
	}

	ImGui::End();
}

void loadSim(const char* config) {
	if (true) {
		try {
			if (config) sim = YarnBall::buildFromJSON(config);
			else sim = YarnBall::buildFromJSON("./configs/cable_work_pattern.json");
		}
		catch (const std::exception& e) {
			printf("Error: %s\n", e.what());
			exit(-1);
		}

		sim->upload();
		printf("Total verts: %d\n", sim->meta.numVerts);
		sim->printErrors = false;
		sim->renderShaded = true;
	}
	else if (false) {
		// Debugging purposes
		constexpr int numVerts = 64;
		sim = new YarnBall::Sim(numVerts);
		const float segLen = 0.002f;

		for (int i = 0; i < 32; i++) {
			vec3 pos = vec3(0.00002f * i * (i - 16) + segLen * 1, -segLen * i, 0);
			sim->verts[i].pos = pos;
			pos.x *= -1;
			sim->verts[i + 32].pos = pos;
		}

		sim->verts[0].invMass = sim->verts[32].invMass = sim->verts[63].invMass = 0;
		sim->verts[31].flags = 0;
		sim->meta.kCollision = 1e-7;
		sim->configure();
		sim->setKBend(1e-8);
		sim->setKStretch(2e-2);
		sim->maxH = 1e-3;
		sim->upload();
		sim->meta.gravity = vec3(-3, -3, 0);
	}

	// Copy initial state for animation.
	initialPos.resize(sim->meta.numVerts);
	initialQ.resize(sim->meta.numVerts - 1);
	for (size_t i = 0; i < sim->meta.numVerts; i++) {
		auto pos = sim->verts[i].pos;
		initialPos[i] = pos;
		initialBounds.absorb(pos);
		if (i < sim->meta.numVerts - 1)
			initialQ[i] = sim->qs[i];
	}

	camera.pos = sim->verts[0].pos;
	camera.minDistance = 0.01f;

	if (!sim) exit(-1);
}

void initScene() {
	Kit::loadDirectory("resources");

	Kit::UBOLight light;
	light.col = vec4(1, 1, 1, 1);
	light.dir = normalize(vec3(1, -2, -1));
	light.hasShadow = true;
	light.type = (int)Kit::KittenLight::DIR;
	Kit::lights.push_back(light);
	light.shadowBias = 0.0001f;
	Kit::shadowDist = 0.5f;

	Kit::ambientLight.col = vec4(0);

	camera.angle = vec2(30, 30);
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
	CLI::App app{ "YarnBall: High performance Cosserat Rods simulation." };

	// string config = "./configs/letterG.json";
	string config = "./configs/cable_work_pattern.json";
	app.add_option("filename", config, "Path to the scene json file")->required(false);

	auto outputOption = app.add_option("-o,--output", exportPath, "Output path prefix (directory must exist). Output file path if last frame only.");
	app.add_option("-n,--nframes", exportLimit, "Number of frames to simulate");

	app.add_flag("--headless", headlessMode, "Run in headless mode (without GUI)");
	app.add_flag("--exit", exitWhenDone, "Exit once all exports are done.");

	app.add_flag("-s", simulate, "Start simulating immediately");
	app.add_flag("-e,--export", exportSim, "Export simulation frames");
	app.add_flag("--exportlast", exportEndFrame, "Export the last frame only");

	app.add_flag("--fiber", exportFiberLevel, "Export as fiber level mesh (slow) instead of obj splines");
	app.add_flag("--bcc", exportBCC, "Export as BCC format instead of obj splines");
	app.add_flag("--twist", scenarioTwist, "Twist animation");
	app.add_flag("--pull", scenarioPull, "Pull animation");

	int exportFPS = 30;
	app.add_option("--fps", exportFPS, "Animation frames per second")->default_val(30);
	EXPORT_DT = 1.f / exportFPS;

	CLI11_PARSE(app, argc, argv);

	if (outputOption->count() && !exportSim) exportSim = true;
	if (exportEndFrame) exportSim = true;

	if (!headlessMode) {
		// Init window and OpenGL
		Kit::initWindow(ivec2(800, 600), "OpenGL Window");

		// Register callbacks
		Kit::getIO().mouseButtonCallback = mouseButtonCallback;
		Kit::getIO().cursorPosCallback = cursorPosCallback;
		Kit::getIO().scrollCallback = scrollCallback;
		Kit::getIO().keyCallback = keyCallback;
		initScene();
	}

	loadSim(config.c_str());

	if (headlessMode) {
		exitWhenDone = true;
		while (true) performSim();
		return 0;
	}

	while (!Kit::shouldClose()) {
		Kit::startFrame();
		if (simulate) performSim();
		renderScene();		// Render
		renderGui();		// GUI Render
		Kit::endFrame();
	}

	Kit::terminate();
	return 0;
}