# ThrustRTC

## Progress

The core infrastructure for handling device-viewable data objects has been constructed.
The following examples shows how to construct and launch general kernels and for-loops,
and how to use them to operate on DVVectors (device viewable GPU vectors).

* test/test_trtc.cpp
* test/test_for.cpp
* python/test/test_trtc.py
* python/test/test_for.py

The following examples shows the use of the ported Thrust algorithms.
(I will populate the code progressively)

* fill
  * test/test_fill.cpp
  * python/test/test_fill.py

* replace
  * test/test_replace.cpp
  * python/test/test_replace.py

* for_each
  * test/test_for_each.cpp
  * python/test/test_for_each.py

## License 

I've decided to license this project under ['"Anti 996" License'](https://github.com/996icu/996.ICU/blob/master/LICENSE)

Basically, you can use the code any way you like unless you are working for a 996 company.

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)



