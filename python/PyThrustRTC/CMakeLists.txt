cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)

set (SRC
PyThrustRTC.cpp
Context.hpp
viewbuf_to_python.hpp
DeviceViewable.hpp
DVVector.hpp
DVConstant.hpp
DVCounter.hpp
DVDiscard.hpp
DVPermutation.hpp
DVReverse.hpp
DVTransform.hpp
fill.hpp
functor.hpp
replace.hpp
for_each.hpp
adjacent_difference.hpp
sequence.hpp
tabulate.hpp
transform.hpp
gather.hpp
scatter.hpp
copy.hpp
swap.hpp
count.hpp
reduce.hpp
equal.hpp
extrema.hpp
inner_product.hpp
transform_reduce.hpp
logical.hpp
scan.hpp
)

set (INCLUDE_DIR 
${PYTHON_INCLUDE_DIRS}
.
..
)


if (WIN32) 
set (DEFINES  ${DEFINES}
-D"_CRT_SECURE_NO_DEPRECATE"  
-D"_SCL_SECURE_NO_DEPRECATE" 
-D"_CRT_SECURE_NO_WARNINGS"
)
else()
add_definitions(-std=c++14)
add_compile_options(-fPIC)
endif()

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})

SET(CMAKE_INSTALL_RPATH ../lib)

add_library (PyThrustRTC SHARED ${SRC})

if (WIN32) 
set_target_properties(PyThrustRTC PROPERTIES SUFFIX ".pyd")
else()
set_target_properties(PyThrustRTC PROPERTIES SUFFIX ".so")
set_target_properties(PyThrustRTC PROPERTIES PREFIX "")
endif()

target_link_libraries(PyThrustRTC ${PYTHON_LIBRARIES} ThrustRTC_static unqlite)


if (WIN32) 
install(TARGETS PyThrustRTC RUNTIME DESTINATION test_python/ThrustRTC)
else()
install(TARGETS PyThrustRTC DESTINATION test_python/ThrustRTC)
endif()

