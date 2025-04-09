# YarnBall Sim: A high performance yarn simulator

A massively parallel GPU implementation of the paper ["Stable Cosserat Rods" SIGGRAPH 2025](https://s2025.siggraph.org/), [Jerry Hsu](https://jerryhsu.io), Tongtong Wang, Kui Wu, and Cem Yuksel. 

This repository contains the source code, model, and paramters for the yarn twisting and yarn letter examples.

[![Example](images/yarnTwist.gif)](https://youtu.be/hmEGLPG1zP0)

^ The above example runs in real time on an NVidia RTX 3090. For the CPU based examples, see the [StableCosseratRods](https://github.com/jerry060599/StableCosseratRods) repo.

## Build
Project templated from [Kitten Engine](https://github.com/jerry060599/KittenEngine/tree/main)

Configured for Windows and Visual Studios. 

**Requires/tested on CUDA 12.8**

**Dependencies using vcpkg**: assimp, eigen3, stb headers, glad, glfw, imgui[opengl3-glad-binding], glm, jsoncpp, cli11

To install these packages:

1. Setup vcpkg https://vcpkg.io/en/getting-started.html

2. Run:
```
vcpkg.exe install glm:x64-windows
vcpkg.exe install eigen3:x64-windows
vcpkg.exe install assimp:x64-windows
vcpkg.exe install glad:x64-windows
vcpkg.exe install freetype:x64-windows
vcpkg.exe install jsoncpp:x64-windows
vcpkg.exe install imgui[core,glfw-binding,opengl3-binding]:x64-windows
vcpkg.exe install cli11:x64-windows
```

**DO NOT INSTALL GLM DIRECTLY FROM THE CURRENT WORKING BRANCH.**
Only install versions taged as stable releases or through vcpkg. 
When in doubt, use glm version ```1.0.1#3```. 

## Usage
This repo contains both a sample CLI and a C++ interface. 
In most cases, it is the easiest to load a scene directly through a provided JSON.
Sample JSON formats can be found in [cable_work_pattern.json](KittenEngine/configs/cable_work_pattern.json) and [letterS.json](KittenEngine/configs/letterS.json).
Curves can be provided with [.bcc](https://www.cemyuksel.com/cyCodeBase/soln/using_bcc_files.html) or .poly files. See [reader.cpp](KittenEngine/YarnBall/io/reader.cpp) for how to load additional formats.

### CLI
The CLI contains some basic functionality for simulating and exporting scenes.

![Gui](images/gui.gif)

By default, the CLI launches a GUI for visualization and additional controls. Left click to move. Right click to pan. Scroll to zoom. Space to pause/unpause. Additionally, the GUI can be turned off with the ```--headless``` flag. 

Run ``` Gui.exe -help``` to see other availible flags.
```
YarnBall: High performance Cosserat Rods simulation.
Usage: C:\Workspace\Yarn\YarnBall\x64\Release\Gui.exe [OPTIONS] [filename]

Positionals:
  filename TEXT               Path to the scene json file

Options:
  -h,--help                   Print this help message and exit
  -o,--output TEXT            Output path prefix (directory must exist). Output file path if last frame only.
  -n,--nframes INT            Number of frames to simulate
  --headless                  Run in headless mode (without GUI)
  -s                          Start simulating immediately
  -e,--export                 Export simulation frames
  --exportlast                Export the last frame only
  --fiber                     Export as fiber level mesh (slow) instead of obj splines
  --bcc                       Export as BCC format instead of obj splines
  --twist                     Twist animation
  --pull                      Pull animation
  --fps INT [30]              Animation frames per second
```

For example, to run the twisting example in the paper, use
```
Gui.exe configs\cable_work_pattern.json --twist -s
```
To export the animation, use
```
Gui.exe configs\cable_work_pattern.json --export --twist -s -n 750
```

### C++ interface
The C++ interface includes everything under the YarnBall namespace and is placed under its separate VS project, "YarnBall".
See [YarnBall.h](KittenEngine/YarnBall/YarnBall.h) for availible functions and [jsonBuilder.cpp](KittenEngine/YarnBall/io/jsonBuilder.cpp) for sample usage.

To use the simulator, first create a ```YarnBall::Sim``` object with the predetermined number of vertices. 
Once created, vertex positions can be populated within the ```sim->verts``` array. 

To create static vertices, simply set ```sim->verts[i].invMass = 0```. In YarnBall, vertex_i is automatically connected with verteix_{i+1} to form segments.
To create breaks into multiple distinct yarn pieces, unset ```VertexFlags::hasNext``` in the bit flag ```sim->verts[i].flags```. 
This can be accomplished by simply setting the entire flag to 0. The rest of the bit flags are automatically populated after ```configure()```. 

Once done, call ```configure()```. ```configure()``` will automatically populate the proper flags, segment orientations, and rest shapes. 
Simulation can then be done by calling ```sim->advance(1/30.f)``` with the desired frame time. 

Note: The data is passed off to the GPU after ```configure()```. As such, any simulation results will need to call ```download()``` for them to be reflected in ```sim->verts```.
Conversely, any changes to ```sim->verts``` needs to call ```upload()``` to take effect. 

For example, the following code creates and simulates 5 seconds of two small pieces of stiff curved yarn. 
```
constexpr int numVerts = 64;
auto sim = new YarnBall::Sim(numVerts);
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

sim->advance(5.f);
```
See [jsonBuilder.cpp](KittenEngine/YarnBall/io/jsonBuilder.cpp) for more availible controls and settings. 

## License
Unless otherwise stated in the file header, the contents of this repository are provided under the following license. Files that specify a different copyright are governed by the terms indicated therein.

Copyright 2025 Jerry Hsu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
