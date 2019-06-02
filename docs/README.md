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

The code does not actually contain any CUDA device code that need to be prebuilt, therefore CUDA SDK is not a requirement at building time.

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

```
$ pip3 install ThrustRTC
```

You will not get the C++ library, headers as well as all the test programs using this installation method.

### Runtime Dependencies

* CUDA driver (up-to-date)
* Shared library of NVRTC 
  
  * Windows: nvrtc64\*.dll, default location: %CUDA_PATH%/bin
  * Linux: libnvrtc.so, default location: /usr/local/cuda/lib64
  
  If the library is not at the default location, you need to call:

  * TRTCContext::set_libnvrtc_path() from C++ or 
  * ThrustRTC.set_libnvrtc_path() from Python

  at run-time to specify the path of the library.

For Python
* Python 3
* numpy
* numba (optional)

## Context Objects

In ThrustRTC, a context object mains a list of headers that need to be included from device code, global constants, and most importantly, a cache of loaded kernels. When user submits a kernel that is already loaded in the current context, it will used directly and will not be compiled and loaded again.

### Creation

In C++, a context object can be created using its default constructor:

```cpp
#include "TRTCContext.h"

int main()
{
	TRTCContext ctx;
	...
}
```

Very similarly in Python:

```python
import ThrustRTC as trtc

ctx = trtc.Context()
```

### Launching a Kernel

User can use a context object to launch a kernel providing the following information:

* Grid Dimensions
* Block Dimensions
* An argument map
  * In C++: a list of AssignedParam structs, each containing a name of a parameter and a pointer to a Device-Viewable Object
  * In Python: a dictionary of Device-Viewable Objects with their names as keys
* Body of the kernel function represented as a string 
* Size of dynamically allocated shared-memory in bytes 

```cpp
#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"

int main()
{
	TRTCContext ctx;

	float test_f[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
	DVVector dvec_in(ctx, "float", 5, test_f);
	DVVector dvec_out(ctx, "float", 5);
	DVFloat dv_k(10.0);

	ctx.launch_kernel({ 1, 1, 1 }, { 128, 1, 1 }, { { "arr_in", &dvec_in }, {"arr_out", &dvec_out }, {"k", &dv_k }},
		"    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
		"    if (idx >= arr_in.size()) return;\n"
		"    arr_out[idx] = arr_in[idx]*k;\n"
		);

	dvec_out.to_host(test_f);
	printf("%f %f %f %f %f\n", test_f[0], test_f[1], test_f[2], test_f[3], test_f[4]);

	return 0;
}

```

```python
import ThrustRTC as trtc
import numpy as np

ctx = trtc.Context()

dvec_in = trtc.device_vector_from_list(ctx, [ 1.0, 2.0, 3.0, 4.0, 5.0 ], 'float')
dvec_out = trtc.device_vector(ctx, 'float', 5)
dv_k = trtc.DVFloat(10.0)

ctx.launch_kernel(1,128, {'arr_in': dvec_in, 'arr_out': dvec_out, 'k': dv_k }, 
	'''
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= arr_in.size()) return;
	arr_out[idx] = arr_in[idx]*k;
	''')

print (dvec_out.to_host())
```

### Launching a Paralleled For Loop

A paralleled for-loop is a special case of a kernel that has only 1 dimension.

User can use a context object to launch a paralleled for-loop providing the following information:

* An iteration range specified by a begin/end pair, or just "n"
* An argument map, the same as launching a kernel
* Name of the iterator variable 
* Body of the for loop represented as a string 

```cpp
#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"

int main()
{
	TRTCContext ctx;

	float test_f[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
	DVVector dvec_in(ctx, "float", 5, test_f);
	DVVector dvec_out(ctx, "float", 5);
	DVFloat dv_k(10.0);

	ctx.launch_for_n(5, { { "arr_in", &dvec_in }, {"arr_out", &dvec_out }, {"k", &dv_k } }, "idx",
		"    arr_out[idx] = arr_in[idx]*k;\n"
	);

	dvec_out.to_host(test_f);
	printf("%f %f %f %f %f\n", test_f[0], test_f[1], test_f[2], test_f[3], test_f[4]);

	return 0;
}
```

```python
import ThrustRTC as trtc
import numpy as np

ctx = trtc.Context()

dvec_in = trtc.device_vector_from_list(ctx, [ 1.0, 2.0, 3.0, 4.0, 5.0 ], 'float')
dvec_out = trtc.device_vector(ctx, 'float', 5)
dv_k = trtc.DVFloat(10.0)

ctx.launch_for_n(5, {'arr_in': dvec_in, 'arr_out': dvec_out, 'k': dv_k }, "idx",
	'''
	arr_out[idx] = arr_in[idx]*k;
	''')

print (dvec_out.to_host())

```


## Device Viewable Objects

## Vectors

## Functors

## Algorithms

