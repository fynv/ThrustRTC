# ThrustRTC

For introduction in Chinese, see [here](https://zhuanlan.zhihu.com/p/62293854).

## Idea

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

ctx = trtc.Context()

A = trtc.device_vector_from_list(ctx, [1, -3, 2, -1], 'int32_t')
trtc.Replace_If(ctx, A, trtc.Functor( ctx, {}, ['x'], '        return x<0;\n'), trtc.DVInt32(0))

# A contains [1, 0, 2, 0]
```

A significant difference between ThrustRTC and Thrust C++ is that ThrustRTC does not include the iterators. 
All operations explicitly work on vectors types. Working-ranges can be specified using begin/end parameters.

## Building the code

### Build-time Dependencies

CUDA SDK is NOT required. At build time, we only need:

* UnQLite source code, as submodule: thirdparty/unqlite
* CMake 3.x
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
* numba (optional)

## Install ThrustRTC for Python from Pypi

Experimental builds for Win64/Linux64 + Python 3.7 are available from [Pypi](https://pypi.org/project/ThrustRTC/)
If your environment matches, you can try:

$ pip3 install ThrustRTC

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

Most Thrust algorithms has been implemented. Exceptions are:

* set-operations: IMO, supporting of "duplicate elements" has made it unnecessarily complicated. 
* bucket-sort: more efficient than merge-sort, but not general. Will wait until there is a performance need.

The following examples shows the use of the implemented Thrust algorithms:

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
  * test/test_adjacent_difference.cpp
  * python/test/test_adjacent_difference.py

* sequence
  * test/test_sequence.cpp
  * python/test/test_sequence.py

* tabulate
  * test/test_tabulate.cpp
  * python/test/test_tabulate.py

* transform
  * test/test_transform.cpp
  * python/test/test_transform.py

* gather
  * test/test_gather.cpp
  * python/test/test_gather.py

* scatter
  * test/test_scatter.cpp
  * python/test/test_scatter.py

* copy
  * test/test_copy.cpp
  * python/test/test_copy.py

* swap
  * test/test_swap.cpp
  * python/test/test_swap.py

* count
  * test/test_count.cpp
  * python/test/test_count.py

* reduce
  * test/test_reduce.cpp
  * python/test/test_reduce.py

* equal
  * test/test_equal.cpp
  * python/test/test_equal.py

* extrema
  * test/test_extrema.cpp
  * python/test/test_extrema.py

* inner_product
  * test/test_inner_product.cpp
  * python/test/test_inner_product.py

* transform_reduce
  * test/test_transform_reduce.cpp
  * python/test/test_transform_reduce.py

* logical
  * test/test_logical.cpp
  * python/test/test_logical.py

* scan
  * test/test_scan.cpp
  * python/test/test_scan.py

* transform_scan
  * test/test_transform_scan.cpp
  * python/test/test_transform_scan.py

* scan_by_key
  * test/test_scan_by_key.cpp
  * python/test/test_scan_by_key.py

* remove
  * test/test_remove.cpp
  * python/test/test_remove.py

* unique
  * test/test_unique.cpp
  * python/test/test_unique.py

* partition
  * test/test_partition.cpp
  * python/test/test_partition.py

* find
  * test/test_find.cpp
  * python/test/test_find.py

* mismatch
  * test/test_mismatch.cpp
  * python/test/test_mismatch.py

* binary_search
  * test/test_binary_search.cpp
  * python/test/test_binary_search.py

* merge
  * test/test_merge.cpp
  * python/test/test_merge.py

* sort
  * test/test_sort.cpp
  * python/test/test_sort.py

Fake-vector classes are provided to provide similar functionality as "fancy-iterators" of Thrust
The following examples shows the use of the fake-vector classes:

* DVConstant (constant_iterator)
  * test/test_constant.cpp
  * python/test/test_constant.py

* DVCounter (counting_iterator)
  * test/test_counter.cpp
  * python/test/test_counter.py

* DVDiscard (discard_iterator) 
  * test/test_discard.cpp
  * python/test/test_discard.py

* DVPermutation (permutation_iterator)
  * test/test_permutation.cpp
  * python/test/test_permutation.py

* DVReverse (reverse_iterator)
  * test/test_reverse.cpp
  * python/test/test_reverse.py

* DVTransform (transform_iterator)
  * test/test_transform_iter.cpp
  * python/test/test_transform_iter.py

* DVZipped (zip_iterator)
  * test/test_zipped.cpp
  * python/test/test_zipped.py

## License 

I've decided to license this project under ['"Anti 996" License'](https://github.com/996icu/996.ICU/blob/master/LICENSE)

Basically, you can use the code any way you like unless you are working for a 996 company.

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)



