# ThrustRTC

For introduction in Chinese, see [here](https://zhuanlan.zhihu.com/p/62293854).

## Idea

The aim of the project is to provide a library of general GPU algorithms, functionally similar to [Thrust](https://github.com/thrust/thrust/), that can be used in non-C++ programming launguages (currently focused on Python).

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

trtc.set_ptx_cache('__ptx_cache__')
ctx = trtc.Context()

A = trtc.device_vector_from_list(ctx, [1, -3, 2, -1], 'int32_t')
trtc.Replace_If(ctx, A, trtc.Functor( {}, ['x'], 'ret', '        ret = x<0;\n'), trtc.DVInt32(0))

# A contains [1, 0, 2, 0]
```

There are several differences between ThrustRTC and Thrust C++:

* ThrustRTC does not include the iterators. All operations explicitly work on the device vectors.
* Functors in ThrustRTC are implemented as "do{...} while(false);" blocks, so "return" is not supported. 
  User need to specify a variable name for the return value and assign to it. "break" is supported though.

In verbose mode we can see the full code of the CUDA kerenel looks like:

```cpp
#define DEVICE_ONLY
#include "DVVector.h"
#include "cstdint"
extern "C" __global__
void saxpy(VectorView<int32_t> _view_vec, int32_t _new_value, size_t _begin, size_t _end)
{
      size_t _idx = threadIdx.x + blockIdx.x*blockDim.x + _begin;
      if (_idx>=_end) return;
      bool ret;
      do{
          auto x = _view_vec[_idx];
          ret = x<0;
      } while(false);
      if (ret) _view_vec[_idx] = _new_value;
}
```

* We may not be able to port all the Thrust algorithms. (but hopefully most of them!)

## Building the code

### Build-time Dependencies

* CMake 3.x
* CUDA toolkit >= 7.0 (CMake needs to find it. Only the driver header cuda.h and cuda.lib/libcuda.so are used)
* C libraries of Python 3 is required to build the Python binding part of the code.

### Building with CMake

```
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_INSTALL_PREFIX=../install
$ make
$ make install
```

You will get the library headers, binaries and examples in the "install" directory.

### Run-time dependencies

* CUDA driver (up-to-date)
* Shared library of NVRTC 
  
  * Windows: nvrtc64\*.dll, default location: %CUDA_PATH%/bin
  * Linux: libnvrtc.so, default location: /usr/local/cuda/lib64
  
  If the library is not at the default location, TRTCContext::set_libnvrtc_path() need to be called at run-time to specify the path of the library.

For Python
* Python 3
* numpy

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

* tabulate
  * test/tabulate.cpp
  * python/test/tabulate.py

## License 

I've decided to license this project under ['"Anti 996" License'](https://github.com/996icu/996.ICU/blob/master/LICENSE)

Basically, you can use the code any way you like unless you are working for a 996 company.

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)



