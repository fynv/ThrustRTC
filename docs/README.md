# ThrustRTC - Quick Start Guide

There will be a long time before ThrustRTC is properly documented. 

ThrustRTC is "multilingual", there is a C++ library at its core, a wrapper layer for Python users, and there will be wrapper layers for C# and Java as well in the future. Therefore, documenting every detail will be a huge effort. However, thanks to [Thrust](https://thrust.github.io/doc/index.html), most of the core logics of the algorithms are already explained. ThrustRTC is following very similar logics as Thrust, so we can just focus on the differences here.

## Introduction

The project ThrustRTC is trying to provide a basic GPU algorithm library that can be used everywhere. 

Most C/C++ libraries are already able to be used "everywhere", as long as the targeting language has a C/C++ interfacing machinism. However, templated libraries are exceptions, they are C++ only. For GPU programming, it is well known that static-polymorphism is preferred over dynamic-polymorphism. Thrust is a CUDA algorithm library that implemented static-polymorphism based on C++ templates. As a downside, it restricted itself to C++. This project tries to overcome this downside of Thrust using NVRTC, through which "static-polymorphism" of GPU programmes can happen at the runtime of the host program.

ThrustRTC provides almost the same algorithms as Thrust, such as scan, sort and reduce, while making these algorithms available to non-C++ launguages.

### Installation

#### Install from Source Code

Source code of ThrustRTC is available at:

[https://github.com/fynv/ThrustRTC](https://github.com/fynv/ThrustRTC)

The code does not actually contain any CUDA code that need to be prebuilt, therefore CUDA SDK is not a requirement at building time.

At build time, you only need:

* UnQLite source code, as submodule: thirdparty/unqlite
* CMake 3.x
* C libraries of Python 3 is required to build the Python binding part of the code.

After cloning the repo from github and resolving the submodules, you can build it with CMake:

```
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_INSTALL_PREFIX=../install
$ make
$ make install
```

You will get the library headers, binaries and examples in the "install" directory.

#### Install ThrustRTC for Python from Pypi

Experimental builds for Win64/Linux64 + Python 3.7 are available from [Pypi](https://pypi.org/project/ThrustRTC/)
If your environment matches, you can try:

$ pip3 install ThrustRTC

## Context Objects

## Device Viewable Objects

## Vectors

## Functors

## Algorithms

