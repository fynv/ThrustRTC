cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)
find_package(CUDA REQUIRED)

set (SRC
PyThrustRTC.cpp
Context.hpp
DeviceViewable.hpp
DVVector.hpp
fill.hpp
functor.hpp
replace.hpp
for_each.hpp
adjacent_difference.hpp
sequence.hpp
tabulate.hpp
transform.hpp
gather.hpp
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
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
else()
add_definitions(-std=c++14)
add_compile_options(-fPIC)
endif()

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})

SET(CMAKE_INSTALL_RPATH ../lib)

cuda_add_library (PyThrustRTC SHARED ${SRC})

if (WIN32) 
set_target_properties(PyThrustRTC PROPERTIES SUFFIX ".pyd")
else()
set_target_properties(PyThrustRTC PROPERTIES SUFFIX ".so")
set_target_properties(PyThrustRTC PROPERTIES PREFIX "")
endif()

target_link_libraries(PyThrustRTC ${PYTHON_LIBRARIES} ThrustRTC_static cuda)


if (WIN32) 
install(TARGETS PyThrustRTC RUNTIME DESTINATION test_python/ThrustRTC)
else()
install(TARGETS PyThrustRTC DESTINATION test_python/ThrustRTC)
endif()

