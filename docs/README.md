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

In ThrustRTC, a Context object mains a list of headers that need to be included from device code, global constants, and most importantly, a cache of loaded kernels. When user submits a kernel that is already loaded in the current context, it will used directly and will not be compiled and loaded again.

### Creation

In C++, a Context object can be created using its default constructor:

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

**Note:** The remaining of this section is about how to write and launch your own kernels using a context object.
If you are only interested in using the built-in algorithms, you can skip them an go ahead from [Device Viewable Objects](#device-viewable-objects).

### Launching a Kernel

User can use a Context object to launch a kernel providing the following information:

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

User can use a Context object to launch a paralleled for-loop providing the following information:

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

### Kernel and For-Loop Objects

Kernel and For-Loop objects can be used to separate their definition from launching.

Note that these objects are just catching the source code, no compilation will happen before they are sent to a context for launching.

A Kernel object can be created given the following:

* Names of parameters
* Body of the kernel function represented as a string 

Then it can be launched given the following:

* The Context object
* Grid Dimensions
* Block Dimensions
* Device Viewable Objects as arguments

Example using Kernel objects:

```cpp
#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"

int main()
{
	TRTCContext ctx;

	TRTC_Kernel ker(
	{ "arr_in", "arr_out", "k" },
	"    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
	"    if (idx >= arr_in.size()) return;\n"
	"    arr_out[idx] = arr_in[idx]*k;\n");

	float test_f[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0 };
	DVVector dvec_in_f(ctx, "float", 5, test_f);
	DVVector dvec_out_f(ctx, "float", 5);
	DVFloat k1(10.0);
	const DeviceViewable* args_f[] = { &dvec_in_f, &dvec_out_f, &k1 };
	ker.launch(ctx, { 1, 1, 1 }, { 128, 1, 1 }, args_f);
	dvec_out_f.to_host(test_f);
	printf("%f %f %f %f %f\n", test_f[0], test_f[1], test_f[2], test_f[3], test_f[4]);

	return 0;
}
```

```python
import ThrustRTC as trtc

ctx = trtc.Context()

kernel = trtc.Kernel(['arr_in', 'arr_out', 'k'],
	'''
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= arr_in.size()) return;
	arr_out[idx] = arr_in[idx]*k;
	''')

dvec_in = trtc.device_vector_from_list(ctx, [ 1.0, 2.0, 3.0, 4.0, 5.0 ], 'float')
dvec_out = trtc.device_vector(ctx, 'float', 5)
dv_k = trtc.DVFloat(10.0)

kernel.launch(ctx, 1,128, [dvec_in, dvec_out, dv_k])
print (dvec_out.to_host())
```

A For-Loop object can be created given the following:

* Names of parameters
* Name of the iterator variable 
* Body of the for loop represented as a string 

Then it can be launched given the following:

* The Context object
* An iteration range specified by a begin/end pair, or just "n"
* Device Viewable Objects as arguments

Example using For-Loop objects:

```cpp
#include <stdio.h>
#include "TRTCContext.h"
#include "DVVector.h"

int main()
{
	TRTCContext ctx;

	TRTC_For f({ "arr_in", "arr_out", "k" }, "idx",
		"    arr_out[idx] = arr_in[idx]*k;\n");

	float test_f[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0 };
	DVVector dvec_in_f(ctx, "float", 5, test_f);
	DVVector dvec_out_f(ctx, "float", 5);
	DVDouble k1(10.0);
	const DeviceViewable* args_f[] = { &dvec_in_f, &dvec_out_f, &k1 };
	f.launch_n(ctx, 5, args_f);
	dvec_out_f.to_host(test_f);
	printf("%f %f %f %f %f\n", test_f[0], test_f[1], test_f[2], test_f[3], test_f[4]);

	return 0;
}
```

```python
import ThrustRTC as trtc
import numpy as np

ctx = trtc.Context()

forLoop = trtc.For(['arr_in','arr_out','k'], "idx",
	'''
	arr_out[idx] = arr_in[idx]*k;
	''')

dvec_in = trtc.device_vector_from_list(ctx, [ 1.0, 2.0, 3.0, 4.0, 5.0 ], 'float')
dvec_out = trtc.device_vector(ctx, 'float', 5)
dv_k = trtc.DVFloat(10.0)

forLoop.launch_n(ctx, 5, [dvec_in, dvec_out, dv_k])

print (dvec_out.to_host())
```

## Device Viewable Objects

Device Viewable Objects are objects that can be used as kernel arguments. All Device Viewable objects are derived from the class DeviceViewable. Device Viewable Objects have their own type information maintained internally. Therefore, you see that the kernel and for-loop parameters do not have their types defined explicitly. You can consider that all parameters are "templated", which means their types are decided by the recieved arguments.

### Basic Types

The following types of Device Viewable Objects can be initalized using values of basic types.

| Name of Class | C++ Type            | Creation (C++)     | Creation (Python)      |
| ------------- | ------------------- | ------------------ | ----------------------- |
| DVInt8        | int8_t              | DVInt8 x(42);      | x = trtc.DVInt8(42)     |
| DVUInt8       | uint8_t             | DVUInt8 x(42);     | x = trtc.DVUInt8(42)    |
| DVInt16       | int16_t             | DVInt16 x(42);     | x = trtc.DVInt16(42)    |
| DVUInt16      | uint16_t            | DVUInt16 x(42);    | x = trtc.DVUInt16(42)   | 
| DVInt32       | int32_t             | DVInt32 x(42);     | x = trtc.DVInt32(42)    |
| DVUInt32      | uint32_t            | DVUInt32 x(42);    | x = trtc.DVUInt32(42)   |
| DVInt64       | int64_t             | DVInt64 x(42);     | x = trtc.DVInt64(42)    | 
| DVUInt64      | uint64_t            | DVUInt64 x(42);    | x = trtc.DVUInt64(42)   | 
| DVFloat       | float               | DVFloat x(42.0f);  | x = trtc.DVFloat(42.0)  |
| DVDouble      | double              | DVDouble x(42.0);  | x = trtc.DVDouble(42.0) |
| DVBool        | bool                | DVBool x(true);    | x = trtc.DVBool(True)   |
| DVSizeT       | size_t              | DVSizeT x(42);     | N/A                     |
| DVChar        | char                | DVChar x(42);      | N/A                     |
| DVSChar       | signed char         | DVSChar x(42);     | N/A                     |
| DVUChar       | unsigned char       | DVUChar x(42);     | N/A                     |
| DVShort       | short               | DVShort x(42);     | N/A                     |
| DVUShort      | unsigned short      | DVUShort x(42);    | N/A                     |
| DVInt         | int                 | DVInt x(42);       | N/A                     |
| DVUInt        | unsigned int        | DVUInt x(42);      | N/A                     |
| DVLong        | long                | DVLong x(42);      | N/A                     |
| DVULong       | unsigned long       | DVULong x(42);     | N/A                     |
| DVLongLong    | long long           | DVLongLong x(42);  | N/A                     |
| DVULongLong   | unsigned long long  | DVULongLong x(42); | N/A                     |

### Tuples

Tuples can be created by combining multiple Device Viewable Objects.
Different from Thrust, in ThrustRTC, elements of a Tuple are accessed by their names, not by indices.
The names of elements need to be specified at the creation of Tuples.

```cpp
DVInt32 d_int(123);
DVFloat d_float(456.0f);
DVTuple d_tuple(ctx, { {"a", &d_int}, {"b",&d_float} });
```

```python
d_int = trtc.DVInt32(123);
d_float = trtc.DVFloat(456.0);
trtc.DVTuple(ctx, {'a': d_int, 'b': d_float}
```

### Advanced Types

Besides the basic types and Tuples, Vectors and Functors are also Device Viewable Objects.
These objects will be explained in separated sections.

Hierarchy of Device Viewable Objects

<img src="DeviceViewable.png" width="500px">

## Vectors

Just like in Thrust, Vector is used as the most important data-container in ThrustRTC.

A difference between ThrustRTC and Thrust is that there are no "iterators" in ThrustRTC.
Vectors are Device Viewable Objects, and algorithms works directly on Vectors.

In Thrust, there are "Fancy Iteractors" like "constant_iterator", "counting_iterator". 
In ThrustRTC, the corresponding functionalities are provided through "Fake Vectors" --
Device Viewable Objects that can be accessed by indices but does not necessarily have
a storage.

### DVVector

#### Creation

In C++ code, a DVVector object can be created given the following:

* The Context object
* Name of the type of elements: it can be anything that CUDA recognizes as a type.
* Number of elements 
* A pointer to a host array as source (optional)

```cpp
TRTCContext ctx;
int hIn[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
DVVector dIn(ctx, "int32_t", 8, hIn);
DVVector dOut(ctx, "int32_t", 8);
```

In Python, there are several ways to create a DVVector object.

* Create from Numpy

```python
ctx = trtc.Context()
harr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
darr = trtc.device_vector_from_numpy(ctx, harr)
```

The supported Numpy dtypes are:

| Numpy dtype | C++ Type |
| ----------- | -------- |
| np.int8     | int8_t   |
| np.uint8    | uint8_t  |
| np.int16    | int16_t  |
| np.uint16   | uint16_t |
| np.int32    | int32_t  |
| np.uint32   | uint32_t |
| np.int64    | int64_t  |
| np.uint64   | uint64_t |
| np.float32  | float    |
| np.float64  | double   |
| np.bool     | bool     |

* Create from Python List

```python
ctx = trtc.Context()
dvec_in = trtc.device_vector_from_list(ctx, [ 1.0, 2.0, 3.0, 4.0, 5.0 ], 'float')
```

The C++ type specified here should be one of the basic types corresponding to a supported Numpy dtype, as listed above.

* Create with Specified Type and Size

```python
ctx = trtc.Context()
dvec_out = trtc.device_vector(ctx, 'float', 5)
```
In this case, the C++ type specified can be any type that CUDA recognizes.

Optionally, a raw C++ pointer to an host array can be passed as the src to copy.

#### to_host()

* The method *to_host()* can be used to copy the content of DVVector to host.

The C++ version needs a host buffer of enough size.

```cpp
TRTCContext ctx;
int hIn[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
DVVector dIn(ctx, "int32_t", 8, hIn);
dIn.to_host(hIn);
```

The Python version returns a Numpy NDArray. The type of elements must be a supported one.

```python
ctx = trtc.Context()
dvec_in = trtc.device_vector_from_list(ctx, [ 1.0, 2.0, 3.0, 4.0, 5.0 ], 'float')
print(dvec_in.to_host())
```

There are optional parameters *begin* and *end* which can be used to specify a range to copy.

### DVVectorAdaptor

DVVectorAdaptor objects are device Vectors with externally managed storage.

In C++ code, user can create a DVVectorAdaptor object just like creating a DVVector, 
passing in pointer of device memory instead of host memory to initialize the object.
The device memory should not be freed while the DVVectorAdaptor object is still being
used.

In Python, DVVectorAdaptor is used by DVNumbaVector to adapt to Numba. A Numba DeviceNDArray
can easily be used as a ThrustRTC recognized Vector like the following code shows:

```python
import ThrustRTC as trtc
import numpy as np
from numba import cuda

ctx = trtc.Context()

nparr = np.array([1, 0, 2, 2, 1, 3], dtype=np.int32)
nbarr = cuda.to_device(nparr)
darr = trtc.DVNumbaVector(ctx, nbarr)
trtc.Inclusive_Scan(ctx, darr, darr)
print(nbarr.copy_to_host())
``` 

### DVConstant

DVConstant is corresponding to *thrust::constant_iterator*. 

A DVConstant object can be created using a constant Device Viewable Object and accessed as a Vector of constant value.

```cpp
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVConstant.h"
#include "copy.h"

int main()
{
	TRTCContext ctx;
	DVConstant src(ctx, DVInt32(123), 10)
	DVVector dst(ctx, "int32_t", 10);
	TRTC_Copy(ctx, src, dst);
	...
}

```

```python
import ThrustRTC as trtc

ctx = trtc.Context()
src = trtc.DVConstant(ctx, trtc.DVInt32(123), 10)
dst = trtc.device_vector(ctx, 'int32_t', 10)
trtc.Copy(ctx, src, dst)

```

### DVCounter

DVCounter is corresponding to *thrust::counting_iterator*. 

A DVCounter object can be created using a constant Device Viewable Object as initial value,
and accessed as a Vector of sequentially changing values.

```cpp
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVCounter.h"
#include "copy.h"

int main()
{
	TRTCContext ctx;
	DVCounter src(ctx, DVInt32(1), 10)
	DVVector dst(ctx, "int32_t", 10);
	TRTC_Copy(ctx, src, dst);
	...
}

```

```python
import ThrustRTC as trtc

ctx = trtc.Context()
src = trtc.DVCounter(ctx, trtc.DVInt32(1), 10)
dst = trtc.device_vector(ctx, 'int32_t', 10)
trtc.Copy(ctx, src, dst)

```

### DVDiscard

DVDiscard is corresponding to *thrust::discard_iterator*. 

A DVDiscard can be created given the element type. It will ignore all value written to it. 
Can be used as a place-holder for an unused output.

### DVPermutation

DVPermutation is corresponding to *thrust::permutation_iterator*. 

A DVPermutation object can be created using a Vector as source and another Vector as indices, 
and then accessed the source Vector in permuted order.

```cpp
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVPermutation.h"
#include "copy.h"

int main()
{
	TRTCContext ctx;

	float hvalues[8] = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f };
	DVVector dvalues(ctx, "float", 8, hvalues);

	int hindices[4] = { 2,6,1,3 };
	DVVector dindices(ctx, "int32_t", 4, hindices);

	DVPermutation src(ctx, dvalues, dindices);
	DVVector dst(ctx, "float", 4);
	
	TRTC_Copy(ctx, src, dst);
	...
}

```

```python
import ThrustRTC as trtc

ctx = trtc.Context()

dvalues = trtc.device_vector_from_list(ctx, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], 'float')
dindices =  trtc.device_vector_from_list(ctx, [2,6,1,3], 'int32_t')
src = trtc.DVPermutation(ctx, dvalues, dindices)
dst = trtc.device_vector(ctx, 'float', 4)

trtc.Copy(ctx, src, dst)

```

### DVReverse

DVReverse is corresponding to *thrust::reverse_iterator*. 

A DVReverse object can be created using a Vector as source and access it in reversed order.

```cpp
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVReverse.h"
#include "copy.h"

int main()
{
	TRTCContext ctx;

	int hvalues[4] = { 3, 7, 2, 5 };
	DVVector dvalues(ctx, "int32_t", 4, hvalues);

	DVReverse src(ctx, dvalues);
	DVVector dst(ctx, "int32_t", 4);
	
	TRTC_Copy(ctx, src, dst);
	...
}

```

```python
import ThrustRTC as trtc

ctx = trtc.Context()

dvalues = trtc.device_vector_from_list(ctx, [3, 7, 2, 5], 'int32_t')
src = trtc.DVReverse(ctx, dvalues)
dst = trtc.device_vector(ctx, 'int32_t', 4)

trtc.Copy(ctx, src, dst)

```

### DVTransform

DVTransform is corresponding to *thrust::transform_iterator*. 

A DVTransform object can be created using a Vector as source and a Functor as operator, 
then access the transformed values of the source Vector.


```cpp
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVTransform.h"
#include "copy.h"

int main()
{
	TRTCContext ctx;

	float hvalues[8] = { 1.0f, 4.0f, 9.0f, 16.0f };
	DVVector dvalues(ctx, "float", 4, hvalues);

	Functor square_root{ ctx, {}, { "x" }, "        return sqrtf(x);\n" };

	DVTransform src(ctx, dvalues, "float", square_root);
	DVVector dst(ctx, "float", 4);
	
	TRTC_Copy(ctx, src, dst);
	...
}

```


```python
import ThrustRTC as trtc

ctx = trtc.Context()

dvalues = trtc.device_vector_from_list(ctx, [1.0, 4.0, 9.0, 16.0], 'float')
src = trtc.DVTransform(ctx, dvalues, 'float', square_root)
dst = trtc.device_vector(ctx, 'float', 4)

trtc.Copy(ctx, src, dst)

```

### DVZipped

DVZipped is corresponding to *thrust::zip_iterator*. 

A DVZipped object can be created using multiple Vectors as inputs and accessed as a Vector consisting 
elements combined from the elements from each of these Vectors.

```cpp
#include "TRTCContext.h"
#include "DVVector.h"
#include "fake_vectors/DVTransform.h"
#include "copy.h"

int main()
{
	TRTCContext ctx;

	int h_int_in[5] = { 0, 1, 2, 3, 4};
	DVVector d_int_in(ctx, "int32_t", 5, h_int_in);
	float h_float_in[5] = { 0.0f, 10.0f, 20.0f, 30.0f, 40.0f };
	DVVector d_float_in(ctx, "float", 5, h_float_in);

	DVVector d_int_out(ctx, "int32_t", 5);
	DVVector d_float_out(ctx, "float", 5);

	DVZipped src(ctx, { &d_int_in, &d_float_in }, { "a", "b" });
	DVZipped dst(ctx, { &d_int_out, &d_float_out }, { "a", "b" });

	TRTC_Copy(ctx, src, dst);
	...
}

```


```python
import ThrustRTC as trtc

ctx = trtc.Context()

d_int_in = trtc.device_vector_from_list(ctx, [0, 1, 2, 3, 4], 'int32_t')
d_float_in = trtc.device_vector_from_list(ctx, [ 0.0, 10.0, 20.0, 30.0, 40.0], 'float')

d_int_out = trtc.device_vector(ctx, 'int32_t', 5)
d_float_out = trtc.device_vector(ctx, 'float', 5)

src = trtc.DVZipped(ctx, [d_int_in, d_float_in], ['a','b'])
dst = trtc.DVZipped(ctx, [d_int_out, d_float_out], ['a','b'])

trtc.Copy(ctx, src, dst)

```


## Functors

## Algorithms

