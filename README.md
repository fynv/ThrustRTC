# ThrustRTC

## The Idea

The aim of the project is to provide a library of general GPU algorithms, functionally similar to [Thrust](https://github.com/thrust/thrust/), that can be used in non-C++ programming launguages that has an interface with C/C++ (Python, C#, JAVA etc).

There are several options to integrate CUDA with a language like Python.

### Writing both host code and device code in the target launguage

A special compiler will be required to translate from the target launguage to GPU executable code. 

Even though the target launguage can be an interpreted launguage, the GPU part of code still has to be compiled
for efficiency. For Python, we will need [Numba](http://numba.pydata.org/numba-doc/0.13/CUDAJit.html) to do the
trick. Numba is great, but when considering building a library, there is a limitation that the resulted library 
will be for Python only. For another lauguage, we will need to find another tool and build the library again.

### Providing precompiled GPU code, accessible through host APIs

This is what we do for most GPU libraries. There are some general limitations:

* Code bloat. Each kernel needs to be compiled for multiple GPU generations. For templated kernels, the number will be multiplied
  with the number of different data types.

* Unextendable. Once a CUDA module is compiled, it will be extremely difficult to insert custom code from outside of the module. 

Thrust uses templates and callback/functors intensively, so the above limitations will be unavoidable.

### Integrate GPU RTC (runtime compilation) with the target launguage

This is the choice of this project. We still write the device code in C++. However, we delay the compilation of device code to runtime.

Per-usecase runtime data-type information and custom code (functors) will be integrated with fixed routines dynamically, 
through a string concatenation procedure, and concrete source code will be generated and compiled as kernels. 

From user's perspective, the usage of this library is quite simlar to using Thrust, although 2 different launguages needed to be used simultaneously. Using _replace_if()_ as an example:

Thrust, C++:

```cpp
#include <vector>
#include <thrust/replace.h>
#include <thrust/device_vector.h>

std::vector<int> hdata({ 1, -3, 2, -1 });
thrust::device_vector<int> A(hdata);
thrust::replace_if(A.begin(), A.end(), [] __device__(int x){ return x < 0; }, 0);

// A contains [1, 0, 2, 0]
```

ThrustRTC, Python (host) + C++ (device):

```python
import ThrustRTC as trtc

A = trtc.device_vector_from_list(ctx, [1, -3, 2, -1], 'int32_t')
trtc.Replace_If(A, trtc.Functor( ctx, {}, ['x'], '        return x<0;\n'), trtc.DVInt32(0))

# A contains [1, 0, 2, 0]
```

A significant difference between ThrustRTC and Thrust C++ is that ThrustRTC does not include the iterators. 
All operations explicitly work on vectors types. Working-ranges can be specified using begin/end parameters.

## Quick Start Guide

[https://fynv.github.io/ThrustRTC/QuickStartGuide.html](https://fynv.github.io/ThrustRTC/QuickStartGuide.html)

## Demos

Using ThrustRTC for histogram calculation and k-means clustering.

[https://fynv.github.io/ThrustRTC/Demo.html](https://fynv.github.io/ThrustRTC/Demo.html)

## License 

I've decided to license this project under ['"Anti 996" License'](https://github.com/996icu/996.ICU/blob/master/LICENSE)

Basically, you can use the code any way you like unless you are working for a 996 company.

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)



