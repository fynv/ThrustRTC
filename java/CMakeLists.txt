cmake_minimum_required (VERSION 3.0)

find_package(JNI REQUIRED)

find_package(Java REQUIRED)
include(UseJava)

set (SRC
JThrustRTC_Native.h
Context.cpp
DeviceViewable.cpp
DVVector.cpp
)

set (INCLUDE_DIR 
${JNI_INCLUDE_DIRS}
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

add_library (JThrustRTC_Native SHARED ${SRC})

target_link_libraries(JThrustRTC_Native ThrustRTC_static unqlite)


if (WIN32) 
install(TARGETS JThrustRTC_Native RUNTIME DESTINATION test_java)
else()
install(TARGETS JThrustRTC_Native DESTINATION test_java)
endif()

##################################################

set(JAVA
Native.java
TRTC.java
DeviceViewable.java
DVInt8.java
DVInt16.java
DVInt32.java
DVInt64.java
DVFloat.java
DVDouble.java
DVVectorLike.java
DVRange.java
DVVector.java
Kernel.java
For.java
)

add_jar(JThrustRTC ${JAVA})
install_jar(JThrustRTC test_java)

##################################################

add_subdirectory(test)
