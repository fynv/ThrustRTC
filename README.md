# ThrustRTC

## Idea

The aim of the project is provide general GPU algorithms, functionally similar to [Thrust](https://github.com/thrust/thrust/),
that can be used in non-C++ programming launguages (currently focused on Python).

There are several options to integrate CUDA into a interpreted language like Python.

* Writing both host code and device code in the interpreted launguage

A special compiler/interpretor is required to translate from the interpreted launguage to GPU executable code.

For Python, we need [Numba](http://numba.pydata.org/numba-doc/0.13/CUDAJit.html), which is great, but only a subset of CUDA features can be utilized.

* Providing precompiled GPU code, accessible through host APIs

This is what we do for most GPU libraries. There are some general limitations:

  * Code bloat. Each kernel needs to be compiled for multiple GPU generations. For templated kernels, the number will be multiplied
    with the number of different data types.

  * Unextendable. Once a CUDA module is compiled, it will be extremely difficult to insert custom code from outside of the module. 

Thrust uses templates and callback/functors intensively, so the above limitations will be unavoidable.

* Integrate GPU RTC (runtime compilation) with the interpreted launguage

This is the choice of this project. We still write the device code in C++. However, we delay the compilation of device code to runtime.

Per-usecase runtime data-type information and custom code (functors) will be integrated with fixed routines dynamically, 
through a string concatenation procedure, and concrete source code will be generated and compiled as kernels. 

From user's perspective, the usage is of this library is quite simlar to using Thrust, although 2 different launguages needed to be used simultaneously. Using _replace_if()_ as an example:

Thrust, C++:

```cpp
#include <thrust/replace.h>
#include <thrust/device_vector.h>

thrust::device_vector<int> A(4);
A[0] =  1;
A[1] = -3;
A[2] =  2;
A[3] = -1;

thrust::replace_if(A.begin(), A.end(), [] __device__ (int x){ return x < 0; }, 0);

// A contains [1, 0, 2, 0]
```

ThrustRTC, Python (host) + C++ (device):

```python
import ThrustRTC as trtc

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

A = trtc.device_vector_from_list(ctx, [1, -3, 2, -1], 'int32_t')

trtc.Replace_If(ctx, A, trtc.Functor( {}, ['x'], 'ret',
'''
         ret = x<0;
'''), trtc.DVInt32(0))

# A contains [1, 0, 2, 0]
```

There are several differences between ThrustRTC and Thrust C++:

* ThrustRTC does not include the iterators. All operations explicitly work on the device vectors.
* Functors in ThrustRTC are implemented as "do{...} while(false);" blocks, so "return" is not supported. 
  User need to specify a variable name for the return value and assign to it. "break" is supported though.
* We may not be able to port all the Thrust algorithms. 

## Progress

The core infrastructure has been built, which includes:

* A context class that manages the included headers, global contants and provides cache of PTX code in both internal and external storages.
* General kernels and for-loops can be launched within a context, given their source-code as strings
* A hierachy of device-viewable classes. The most important one is DVVectors, device-viewable GPU vectors. These objects can be created and managed by host and passed to device. 

The following examples shows the basic usage of the core infrastructure:

* test/test_trtc.cpp
* test/test_for.cpp
* python/test/test_trtc.py
* python/test/test_for.py

Thrust algorithms are being ported progressively.
The following examples shows the use of the ported Thrust algorithms:
* fill
  * test/test_fill.cpp
  * python/test/test_fill.py

* replace
  * test/test_replace.cpp
  * python/test/test_replace.py

* for_each
  * test/test_for_each.cpp
  * python/test/test_for_each.py

* adjacent_difference
  * test/adjacent_difference.cpp
  * python/test/adjacent_difference.py

* sequence
  * test/sequence.cpp
  * python/test/sequence.py

## License 

I've decided to license this project under ['"Anti 996" License'](https://github.com/996icu/996.ICU/blob/master/LICENSE)

Basically, you can use the code any way you like unless you are working for a 996 company.

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)



