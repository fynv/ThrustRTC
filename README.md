# ThrustRTC

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



