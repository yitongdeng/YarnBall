# YarnBall Sim: A yarn level cloth simulator (Stable Cosserat Rods)

A massively parallel GPU implementation of the paper ["Stable Cosserat Rods", SIGGRAPH 2025](https://s2025.siggraph.org/). 

This repository contains the source code, model, and paramters for the yarn twisting example.

For the CPU based examples, see the [StableCosseratRods](https://github.com/jerry060599/StableCosseratRods) repo.

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

### CLI
The CLI contains some basic functionality for simulating and exporting scenes.
Run ``` Gui.exe -help``` to see availible flags.
For example, to run the twisting example in the paper, use
```
Gui.exe configs\cable_work_pattern.json --twist -s
```
To export the animation, use
```
Gui.exe configs\cable_work_pattern.json --export --twist -s -n 750
```

### C++ interface


## License
Unless otherwise stated in the file header, the contents of this repository are provided under the following license. Files that specify a different copyright are governed by the terms indicated therein.

Copyright 2025 Jerry Hsu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
