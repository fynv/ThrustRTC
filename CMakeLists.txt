cmake_minimum_required (VERSION 3.0)

find_package(PythonLibs 3 REQUIRED)
find_package(CUDA REQUIRED)

set(LIB_SOURCES
DeviceViewable.cpp
TRTCContext.cpp
DVVector.cpp
internal/md5.c
)

set(LIB_HEADERS
DeviceViewable.h
TRTCContext.h
DVVector.h
TRTC_api.h
)

set(INTERNAL_HEADERS
internal/Timing.h
internal/md5.h
)

set (INCLUDE_DIR 
${PYTHON_INCLUDE_DIRS}
.
./internal
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
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif()

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})

SET(CMAKE_INSTALL_RPATH ../lib)

cuda_add_library (ThrustRTC SHARED ${LIB_SOURCES} ${LIB_HEADERS} ${INTERNAL_HEADERS})

add_executable(PackHeaders internal/pack_headers.cpp)
add_custom_target(Run_PackHeaders
COMMAND PackHeaders
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
COMMENT "Running PackHeaders in ${CMAKE_CURRENT_SOURCE_DIR}"
SOURCES internal/pack_headers.cpp ${LIB_HEADERS}
)
add_dependencies(Run_PackHeaders PackHeaders)
add_dependencies(ThrustRTC Run_PackHeaders)

if (WIN32) 
target_compile_definitions(ThrustRTC PUBLIC THRUST_RTC_DLL_EXPORT)
endif()

target_link_libraries(ThrustRTC cuda nvrtc)

cuda_add_executable(test_trtc test/test_trtc.cpp)

if (WIN32) 
target_compile_definitions(test_trtc PUBLIC THRUST_RTC_DLL_IMPORT)
endif()

target_link_libraries(test_trtc ThrustRTC)

install(FILES ${LIB_HEADERS} DESTINATION include)
install(TARGETS test_trtc RUNTIME DESTINATION test)

if (WIN32) 
install(TARGETS ThrustRTC 
		RUNTIME DESTINATION test
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib 
		)
else()
install(TARGETS ThrustRTC DESTINATION lib)
endif()
