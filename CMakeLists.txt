cmake_minimum_required (VERSION 3.0)

find_package(CUDA REQUIRED)

set(LIB_SOURCES
TRTCContext.cpp
DVVector.cpp
internal/md5.c
for.cpp
fill.cpp
replace.cpp
for_each.cpp
adjacent_difference.cpp
sequence.cpp
)

set(LIB_HEADERS
TRTC_api.h
DeviceViewable.h
TRTCContext.h
DVVector.h
for.h
functor.h
fill.h
replace.h
for_each.h
adjacent_difference.h
sequence.h
)

set(INTERNAL_HEADERS
internal/Timing.h
internal/md5.h
)

set (INCLUDE_DIR 
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
target_compile_definitions(ThrustRTC PRIVATE THRUST_RTC_DLL_EXPORT)
endif()

target_link_libraries(ThrustRTC cuda nvrtc)

cuda_add_executable(test_trtc test/test_trtc.cpp)
cuda_add_executable(test_for test/test_for.cpp)
cuda_add_executable(test_fill test/test_fill.cpp)
cuda_add_executable(test_replace test/test_replace.cpp)
cuda_add_executable(test_for_each test/test_for_each.cpp)
cuda_add_executable(test_adjacent_difference test/test_adjacent_difference.cpp)
cuda_add_executable(test_sequence test/test_sequence.cpp)

if (WIN32) 
target_compile_definitions(test_trtc PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_for PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_fill PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_replace PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_for_each PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_adjacent_difference PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_sequence PUBLIC THRUST_RTC_DLL_IMPORT)
endif()

target_link_libraries(test_trtc ThrustRTC)
target_link_libraries(test_for ThrustRTC)
target_link_libraries(test_fill ThrustRTC)
target_link_libraries(test_replace ThrustRTC)
target_link_libraries(test_for_each ThrustRTC)
target_link_libraries(test_adjacent_difference ThrustRTC)
target_link_libraries(test_sequence ThrustRTC)

install(FILES ${LIB_HEADERS} DESTINATION include)

if (WIN32) 
install(TARGETS ThrustRTC 
		RUNTIME DESTINATION test
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib 
		)
else()
install(TARGETS ThrustRTC DESTINATION lib)
endif()

install(TARGETS 
test_trtc 
test_for 
test_fill 
test_replace 
test_for_each 
test_adjacent_difference 
test_sequence
RUNTIME DESTINATION test)

install(DIRECTORY DESTINATION test/__ptx_cache__)

add_subdirectory(python/PyThrustRTC)
add_subdirectory(python)