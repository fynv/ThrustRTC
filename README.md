# ThrustRTC

The aim of the project is to provide a library of general GPU algorithms, functionally similar to [Thrust](https://github.com/thrust/thrust/), that can be used in non-C++ programming launguages that has an interface with C/C++ (Python, C#, JAVA etc).

This projects uses a new CUDA programming paradigm: NVRTC + dynamic-instantiation, as an alternative to the well
establish "CUDA runtime + static compilation + templates" paradigm.

Click [here](https://fynv.github.io/ProgrammingGPUAcrossTheLaunguageBoundaries.html) to learn more about the new paradigm.

## Using ThrustRTC in different languages

The usage of this library is quite simlar to using Thrust, except that you can use it Python, C# and JAVA, and CUDA SDK is not required.

Thrust, C++:

```cpp
#include <vector>
#include <thrust/replace.h>
#include <thrust/device_vector.h>

std::vector<int> hdata({ 1, 2, 3, 1, 2  });
thrust::device_vector<int> A(hdata);
thrust::replace(A.begin(), A.end(), 1, 99);

// A contains { 99, 2, 3, 99, 2}
```

ThrustRTC, in C++:
```cpp
#include "TRTCContext.h"
#include "DVVector.h"
#include "replace.h"

int hdata[5] = { 1,2,3,1,2 };
DVVector A("int32_t", 5, hdata);
TRTC_Replace(A, DVInt32(1), DVInt32(99));

// A contains { 99, 2, 3, 99, 2}
```

ThrustRTC, in Python:

```python
import ThrustRTC as trtc

A = trtc.device_vector_from_list([1, 2, 3, 1, 2], 'int32_t')
trtc.Replace(A, trtc.DVInt32(1), trtc.DVInt32(99))

# A contains [99, 2, 3, 99, 2]
```

ThrustRTC, in C#:
```cs
using ThrustRTCSharp;

DVVector A = new DVVector(new int[] { 1, 2, 3, 1, 2 });
TRTC.Replace(A, new DVInt32(1), new DVInt32(99));

// A contains { 99, 2, 3, 99, 2}
```

ThrustRTC, in JAVA:
```java
import JThrustRTC.*;

DVVector vec = new DVVector(new int[] { 1, 2, 3, 1, 2 });
TRTC.Replace(vec, new DVInt32(1), new DVInt32(99));

// A contains { 99, 2, 3, 99, 2}
```

A significant difference between ThrustRTC and Thrust is that ThrustRTC does not include the iterators. 
All operations explicitly work on vectors types. There are adaptive objects that can be used to map to 
a sub-range of a vector instead of using the whole vector.

## Quick Start Guide

[Quick Start Guide - for Python users](https://fynv.github.io/ThrustRTC/QuickStartGuide.html)

[Quick Start Guide - for C# users](https://fynv.github.io/ThrustRTC/QuickStartGuide_cs.html)

[Quick Start Guide - for JAVA users](https://fynv.github.io/ThrustRTC/QuickStartGuide_java.html)


## Demos

Using ThrustRTC for histogram calculation and k-means clustering.

[https://fynv.github.io/ThrustRTC/Demo.html](https://fynv.github.io/ThrustRTC/Demo.html)

## License 

I've decided to license this project under ['"Anti 996" License'](https://github.com/996icu/996.ICU/blob/master/LICENSE)

Basically, you can use the code any way you like unless you are working for a 996 company.

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)



